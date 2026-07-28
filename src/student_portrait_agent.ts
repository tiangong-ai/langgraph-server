import { PutObjectCommand, S3Client } from '@aws-sdk/client-s3';
import { Annotation, StateGraph } from '@langchain/langgraph';
import { ChatOpenAI } from '@langchain/openai';
import axios from 'axios';
import FormData from 'form-data';
import { basename } from 'node:path';
import { z } from 'zod';
import './instrumentation.js';

const openai_api_key = process.env.OPENAI_API_KEY ?? '';
const openai_chat_model =  process.env.OPENAI_CHAT_MODEL_REASONING ?? '';
const openai_chat_model_mini = process.env.OPENAI_CHAT_MODEL_MINI ?? '';
const mineru_base_url = process.env.MINERU_BASE_URL ?? '';
const mineru_api_key = process.env.MINERU_API_KEY ?? '';
const mineru_vision_provider = process.env.MINERU_VISION_PROVIDER ?? '';
const mineru_vision_model = process.env.MINERU_VISION_MODEL ?? '';
const student_portrait_bucket = process.env.STUDENT_PORTRAIT_BUCKET ?? '';
const student_portrait_bucket_region = process.env.STUDENT_PORTRAIT_BUCKET_REGION ?? '';
const student_portrait_bucket_access_key = process.env.STUDENT_PORTRAIT_BUCKET_ACCESS_KEY ?? '';
const student_portrait_bucket_secret_key = process.env.STUDENT_PORTRAIT_BUCKET_SECRET_KEY ?? '';

let cachedS3Client: S3Client | undefined;

function getS3Client() {
  if (!cachedS3Client) {
    const hasStaticCredentials =
      student_portrait_bucket_access_key.length > 0 &&
      student_portrait_bucket_secret_key.length > 0;

    cachedS3Client = new S3Client({
      region: student_portrait_bucket_region || undefined,
      credentials: hasStaticCredentials
        ? {
            accessKeyId: student_portrait_bucket_access_key,
            secretAccessKey: student_portrait_bucket_secret_key,
          }
        : undefined,
    });
  }
  return cachedS3Client;
}

const gradeEnum = z.enum(['A', 'B', 'C', 'D', 'E']);

const scoreSchema = z.object({
  // AI 生成: 0-100 的数值型总分
  total_score: z.number().min(0).max(100).describe('0-100之间的综合得分'),
  // AI 生成: 基于 total_score 的等级（A-E）
  grade: gradeEnum.describe('根据总分映射的等级 A-E'),
  // AI 生成: 对得分的简要解释（必须基于输入证据）
  interpretation: z.string().describe('简要说明评分依据、整体表现'),
});

// 本地截断/摘要长度常量（运行时内部使用）
// 说明：保护 LLM 上下文长度，避免因输入过长导致请求失败或被中途取消（aborted）。
const TEXT_SUMMARY_MAX_LENGTH = 50000; // 针对传入 LLM 的 JSON/文本做保护性截断

// StudentInfo: 学生基础信息结构
// 来源: 外部输入（由调用方/上层服务提供）。字段均为可选，具体使用场景可酌情要求必填项。
type StudentInfo = {
  grade?: string;
  major_background?: string;
  grasp_level?: string;
  major?: string;
  user_id?: number;
};

async function uploadPortraitMarkdownToS3(
  markdown: string,
  studentInfo?: StudentInfo,
): Promise<string | undefined> {
  const userId = studentInfo?.user_id;
  if (!userId) {
    return undefined;
  }

  const bucket = student_portrait_bucket;
  const objectKey = `${userId}.md`;

  const client = getS3Client();
  const command = new PutObjectCommand({
    Bucket: bucket,
    Key: objectKey,
    Body: Buffer.from(markdown, 'utf-8'),
    ContentType: 'text/markdown; charset=utf-8',
  });

  try {
    const res = await client.send(command);
    return res.VersionId;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(
      `Failed to upload student portrait markdown to S3 for user ${objectKey}: ${message}`,
    );
  }
}

// AnswerDetail: 单题详情（来源通常为学习计划或练习数据，属于外部输入）
type AnswerDetail = {
  stem?: string;
  opts?: Array<string | number>;
  ans?: Array<string | number>;
  user_ans?: Array<string | number>;
  score?: number | string;
  analysis?: string;
  feedback?: string;
  knowledge_points?: string[];
};

// PlanDetailNode: 学习计划/知识节点（外部输入），可递归包含子节点与答题记录
type PlanDetailNode = {
  node_id?: string;
  node_name?: string;
  conversations?: string[];
  answer_details?: AnswerDetail[];
  plan_detail?: PlanDetailNode[];
};

// ArtifactAnalysis: 对作品的解析结果，主要由外部解析服务（mineru）或本地处理生成
type ArtifactAnalysis = {
  file_url: string;
  mineru_payload_length?: number;
  summary: string;
  errors?: string[];
};

// portraitSchema: 最终学生画像的结构定义（AI 生成的输出应符合此 schema）
const portraitSchema = z.object({
  overview: z
    .string()
    .describe('对学生在该主题下整体学习状态的高度概括，点出过程与最终表现的核心结论。'),
  process_assessment: z.object({
    score: scoreSchema.describe('过程考查的整体评分结果，需结合学习过程表现给出0-100分与等级'),
    knowledge_structure: z
      .array(
        z.object({
          node_id: z.string().nullable(),
          node_name: z.string().describe('知识节点或子主题名称'),
          mastery_level: z.string().describe('该知识点的掌握度评价，可包含等级与简短说明。'),
          strengths: z.array(z.string()).describe('该知识点表现良好的方面').default([]),
          issues: z.array(z.string()).describe('存在的问题、错误类型或理解薄弱点').default([]),
          recommended_actions: z
            .array(z.string())
            .describe('针对该知识点的改进建议或练习方向')
            .default([]),
        }),
      )
      .describe('针对知识体系中每个节点的掌握情况分析')
      .default([]),
    metrics: z.object({
      mastery: z.object({
        level: z.string().describe('综合掌握度评价结果，需明确等级并说明其含义'),
        interpretation: z.string().describe('对掌握度评价的解释与背后原因'),
      }),
      stability: z.object({
        level: z.string().describe('稳定性水平，如稳定、波动较大等，需要有定性或定量描述'),
        interpretation: z.string().describe('稳定性表现的解析与潜在原因'),
        fluctuations: z.array(z.string()).describe('识别到的波动或不稳定环节').default([]),
      }),
      transfer: z.object({
        level: z.string().describe('迁移度水平，如较强/一般/较弱等，需要明确评价标准'),
        interpretation: z.string().describe('对迁移能力的分析，包括在综合或跨知识点题目中的表现'),
        gaps: z.array(z.string()).describe('迁移过程中暴露出的不足或卡点').default([]),
      }),
    }),
    next_steps: z.object({
      priorities: z.array(z.string()).describe('下一阶段最重要的关注重点，按优先级排序'),
      recommended_practices: z.array(z.string()).describe('具体的练习或学习任务建议').default([]),
      monitoring_indicators: z.array(z.string()).describe('建议持续追踪的指标或检查点').default([]),
    }),
    communication_notes: z
      .array(z.string())
      .describe('与学生或家长沟通时的注意事项、鼓励点或风险提示')
      .default([]),
  }),
  performance_assessment: z.object({
    overall_score: scoreSchema.describe('最终表现的综合评分结果，需给出0-100分与等级'),
    overall_summary: z
      .string()
      .describe('对最终作品整体表现性评价的综合总结，需突出关键结论与等级判断'),
    artifacts: z
      .array(
        z.object({
          file_url: z.string().describe('作品文件链接'),
          ratings: z.object({
            content_quality: z.object({
              score: z.number().min(0).max(100),
              level: z.string(),
              comments: z.string(),
            }),
            thinking_innovation: z.object({
              score: z.number().min(0).max(100),
              level: z.string(),
              comments: z.string(),
            }),
            expression_presentation: z.object({
              score: z.number().min(0).max(100),
              level: z.string(),
              comments: z.string(),
            }),
            norms_reflection: z.object({
              score: z.number().min(0).max(100),
              level: z.string(),
              comments: z.string(),
            }),
          }),
          total_score: z.number().min(0).max(100),
          grade: gradeEnum,
          strengths: z.array(z.string()).default([]),
          issues: z.array(z.string()).default([]),
          improvement_actions: z.array(z.string()).default([]),
          // evidence: z.array(z.string()).default([]),
        }),
      )
      .default([]),
    monitoring_recommendations: z.array(z.string()).default([]),
  }),
});

// 衍生 schema：用于对不同阶段/环节的 AI 输出进行结构化约束
const processAssessmentSchema = portraitSchema.shape.process_assessment;
const performanceAssessmentSchema = portraitSchema.shape.performance_assessment;
const portraitOverviewSchema = z.object({
  overview: z.string().describe('综合过程考查与最终表现后对学生水平与改进重点的 2-3 句总结。'),
});
const portraitMarkdownSchema = z.object({
  markdown: z
    .string()
    .describe(
      '最终呈现给学生的 Markdown 画像，应包含过程考查、最终表现、下一步建议等结构化部分。',
    ),
});

// 注意：这些类型大部分对应的是 AI/模型生成的内容（由 structured LLM 输出并用这些 schema 验证）。
type StudentPortrait = z.infer<typeof portraitSchema>;
type ProcessAssessment = z.infer<typeof processAssessmentSchema>;
type PerformanceAssessment = z.infer<typeof performanceAssessmentSchema>;

// chainState: 状态图（StateGraph）的根注解定义
// 说明每个字段的来源（外部输入 / 本地生成 / AI 生成）及必填性：
// - 输入（外部）: theme、student_info、plan_detail、final_artifacts 等由外部提交的数据
// - 本地生成: plan_summary（由 summarizePlanDetail 生成）、artifact_summary（由 buildArtifactSummary 生成）、portrait_json 等
// - AI（模型）生成: process_assessment、performance_assessment、portrait、portrait_markdown（由相应的 assess*/generate/render 节点生成）
const chainState = Annotation.Root({
  // 学习主题/课程（外部输入，建议提供以提高提示质量）
  theme: Annotation<string>(),
  // 学生基础信息（外部输入，可选）
  student_info: Annotation<StudentInfo>(),
  // 学习计划与答题记录（外部输入，可选）
  plan_detail: Annotation<PlanDetailNode[]>(),
  // 学习计划摘要（本地/工作流生成，由 summarizePlanDetail 填充）
  plan_summary: Annotation<string>(),
  // 最终作品文件链接列表（外部输入），每个元素应为可访问的 file_url
  final_artifacts: Annotation<string[]>(),
  // 过程考查（AI 生成，assessProcess 产出，结构受 processAssessmentSchema 约束）
  process_assessment: Annotation<ProcessAssessment>(),
  // 最终表现评分（AI 生成，assessPerformance 产出，结构受 performanceAssessmentSchema 约束）
  performance_assessment: Annotation<PerformanceAssessment>(),
  // 作品解析（由 analyzeArtifacts 调用外部文档拆解服务或本地处理生成；可累加）
  artifact_analysis: Annotation<ArtifactAnalysis[]>({
    reducer: (x, y) => {
      const base = Array.isArray(x) ? x : [];
      if (!y) {
        return base;
      }
      return base.concat(y);
    },
    default: () => [],
  }),
  // 作品解析汇总（本地生成，基于 artifact_analysis）
  artifact_summary: Annotation<string>(),
  // 最终学生画像结构（由 generatePortrait/模型生成）
  portrait: Annotation<StudentPortrait>(),
  // portrait 的 JSON 字符串（本地生成，便于存储或传输）
  portrait_json: Annotation<string>(),
  // 最终 Markdown 输出（由 renderMarkdown 生成，AI 帮助排版，但内容基于 portrait）
  portrait_markdown: Annotation<string>(),
});

function normalizeValue(value: string | number | null | undefined): string {
  if (value === null || value === undefined) {
    return '';
  }
  return value.toString().trim();
}

function truncateText(value: string, maxLength = TEXT_SUMMARY_MAX_LENGTH): string {
  if (!value) {
    return '';
  }
  if (value.length <= maxLength) {
    return value;
  }
  return `${value.slice(0, maxLength)}\n...[已截断，原始长度 ${value.length} 字符]`;
}

function computeLocalStats(node?: PlanDetailNode) {
  let total = 0;
  let answered = 0;
  let correct = 0;

  node?.answer_details?.forEach((detail) => {
    const standard = (detail.ans ?? []).map(normalizeValue).filter(Boolean);
    const user = (detail.user_ans ?? []).map(normalizeValue).filter(Boolean);
    if (standard.length > 0) {
      total += 1;
    }
    if (user.length > 0) {
      answered += 1;
    }
    if (standard.length > 0 && user.length > 0) {
      const standardSet = new Set(standard);
      const userSet = new Set(user);
      const isExactMatch =
        standardSet.size === userSet.size &&
        Array.from(standardSet).every((entry) => userSet.has(entry));
      if (isExactMatch) {
        correct += 1;
      }
    }
  });

  return { total, answered, correct };
}

function aggregateStats(nodes?: PlanDetailNode[]): {
  total: number;
  answered: number;
  correct: number;
} {
  return (nodes ?? []).reduce(
    (acc, node) => {
      const local = computeLocalStats(node);
      acc.total += local.total;
      acc.answered += local.answered;
      acc.correct += local.correct;
      const child = aggregateStats(node.plan_detail);
      acc.total += child.total;
      acc.answered += child.answered;
      acc.correct += child.correct;
      return acc;
    },
    { total: 0, answered: 0, correct: 0 },
  );
}

function summarizePlanNodes(
  nodes: PlanDetailNode[] | undefined,
  depth = 0,
  path: string[] = [],
): string[] {
  if (!nodes || nodes.length === 0) {
    return [];
  }

  return nodes.flatMap((node) => {
    const indent = '  '.repeat(depth);
    const displayName = node.node_name ?? `未命名节点${node.node_id ?? ''}`;
    const currentPath = [...path, displayName].join(' > ');
    const lines: string[] = [];

    lines.push(`${indent}- 节点: ${displayName}${node.node_id ? ` (ID: ${node.node_id})` : ''}`);
    lines.push(`${indent}  路径: ${currentPath}`);

    const localStats = computeLocalStats(node);
    if (localStats.total > 0 || localStats.answered > 0) {
      const accuracy =
        localStats.total > 0
          ? `${((localStats.correct / localStats.total) * 100).toFixed(1)}%`
          : '无可比对';
      lines.push(
        `${indent}  答题统计: 标准题目${localStats.total}题，已作答${localStats.answered}题，参考答案完全匹配${localStats.correct}题，正确率${accuracy}`,
      );
    }

    if (node.conversations?.length) {
      lines.push(`${indent}  提问记录: ${node.conversations.join('； ')}`);
    }

    node.answer_details?.forEach((detail, index) => {
      lines.push(`${indent}  练习${index + 1}: ${detail.stem ?? '未提供题干'}`);
      if (detail.opts?.length) {
        lines.push(`${indent}    选项: ${detail.opts.map(normalizeValue).join(' | ')}`);
      }
      if (detail.ans?.length) {
        lines.push(`${indent}    参考答案: ${detail.ans.map(normalizeValue).join(' | ')}`);
      }
      if (detail.user_ans?.length) {
        lines.push(`${indent}    学生作答: ${detail.user_ans.map(normalizeValue).join(' | ')}`);
      }
      if (detail.score !== undefined && detail.score !== null && detail.score !== '') {
        lines.push(`${indent}    得分或评价: ${detail.score}`);
      }
      if (detail.analysis) {
        lines.push(`${indent}    解析: ${detail.analysis}`);
      }
      if (detail.feedback) {
        lines.push(`${indent}    反馈: ${detail.feedback}`);
      }
      if (detail.knowledge_points?.length) {
        lines.push(`${indent}    涉及知识点: ${detail.knowledge_points.join(' | ')}`);
      }
    });

    if (node.plan_detail?.length) {
      lines.push(...summarizePlanNodes(node.plan_detail, depth + 1, [...path, displayName]));
    }

    return lines;
  });
}

function buildPlanSummary(nodes?: PlanDetailNode[]): string {
  if (!nodes || nodes.length === 0) {
    return '未提供学习计划与作答数据。';
  }

  const totals = aggregateStats(nodes);
  const accuracy =
    totals.total > 0 ? `${((totals.correct / totals.total) * 100).toFixed(1)}%` : '无可比对';

  const header = `整体答题统计: 标准题目${totals.total}题，已作答${totals.answered}题，参考答案完全匹配${totals.correct}题，正确率${accuracy}`;
  const detailLines = summarizePlanNodes(nodes);
  return [header, ...detailLines].join('\n');
}

function formatStudentInfo(info?: StudentInfo): string {
  if (!info) {
    return '未提供学生信息。';
  }

  const parts = [
    info.grade ? `年级: ${info.grade}` : null,
    info.major ? `专业: ${info.major}` : null,
    info.major_background ? `专业背景: ${info.major_background}` : null,
    // info.grasp_level ? `自评掌握程度: ${info.grasp_level}` : null,
  ].filter(Boolean);

  return parts.length > 0 ? parts.join('； ') : '未提供详细的学生背景。';
}

async function summarizePlanDetail(state: typeof chainState.State) {
  const plan_summary = buildPlanSummary(state.plan_detail);
  return { plan_summary };
}

type MineruTextChunk = {
  text?: string;
  page_number?: number;
};

type MineruWithImagesResponse = {
  result?: MineruTextChunk[];
};

function extractFilenameFromUrl(fileUrl: string): string {
  try {
    const parsed = new URL(fileUrl);
    if (parsed.pathname) {
      const decoded = decodeURIComponent(parsed.pathname);
      const name = basename(decoded);
      if (name) {
        return name;
      }
    }
  } catch {
    // Ignore malformed URL and fall back to default name
  }
  return 'artifact';
}

type DownloadedArtifact = {
  buffer: Buffer;
  filename: string;
  contentType: string;
};

async function downloadArtifactFile(fileUrl: string): Promise<DownloadedArtifact> {
  const response = await axios.get<ArrayBuffer>(fileUrl, {
    responseType: 'arraybuffer',
    timeout: 120000,
  });

  const buffer = Buffer.from(response.data);
  const filename = extractFilenameFromUrl(fileUrl);
  const headerValue = response.headers?.['content-type'];
  const contentType =
    typeof headerValue === 'string' && headerValue.trim().length > 0
      ? headerValue.split(';')[0].trim()
      : 'application/octet-stream';

  return {
    buffer,
    filename,
    contentType,
  };
}

async function callMineruWithImages(file: DownloadedArtifact): Promise<MineruWithImagesResponse> {
  if (!mineru_base_url) {
    throw new Error('文档拆解服务未配置，请设置 MINERU_BASE_URL');
  }

  const form = new FormData();
  form.append('file', file.buffer, {
    filename: file.filename,
    contentType: file.contentType,
  });
  if (mineru_vision_provider.trim().length > 0) {
    form.append('provider', mineru_vision_provider);
  }
  if (mineru_vision_model.trim().length > 0) {
    form.append('model', mineru_vision_model);
  }

  const headers: Record<string, string> = {
    ...form.getHeaders(),
    Accept: 'application/json',
  };

  if (mineru_api_key) {
    headers.Authorization = `Bearer ${mineru_api_key}`;
  }

  const response = await axios.post<MineruWithImagesResponse>(mineru_base_url, form, {
    headers,
    timeout: 300000,
    maxBodyLength: Infinity,
    params: {
      chunk_type: 'true',
      pretty: 'true',
    },
  });

  return response.data;
}

function buildMineruSummary(payload: MineruWithImagesResponse): string {
  const chunks = Array.isArray(payload?.result) ? payload.result : [];
  if (chunks.length === 0) {
    return '';
  }

  const lines = chunks
    .map((chunk) => {
      const text = typeof chunk?.text === 'string' ? chunk.text.trim() : '';
      if (!text) {
        return '';
      }
      const pageNumberRaw = chunk?.page_number;
      const pageNumber =
        typeof pageNumberRaw === 'number'
          ? pageNumberRaw
          : Number.isFinite(Number(pageNumberRaw))
            ? Number(pageNumberRaw)
            : undefined;
      const pageLabel = pageNumber && pageNumber > 0 ? `第${pageNumber}页` : '未注明页码';
      return `[${pageLabel}] ${text}`;
    })
    .filter((line) => line.length > 0);

  return lines.length > 0 ? lines.join('\n\n') : '';
}

function buildArtifactSummary(analyses: ArtifactAnalysis[]): string {
  if (analyses.length === 0) {
    return '未提供最终作品或暂未完成解析。';
  }

  return analyses
    .map((analysis) => {
      const errorLine =
        analysis.errors && analysis.errors.length > 0
          ? `解析异常: ${analysis.errors.join('； ')}`
          : '';

      return [
        `解析摘要:\n${analysis.summary || '未获取到有效内容。'}`,
        errorLine,
        // payloadLine,
      ]
        .filter(Boolean)
        .join('\n');
    })
    .join('\n\n');
}

async function analyzeArtifacts(state: typeof chainState.State) {
  const artifacts = state.final_artifacts ?? [];

  if (artifacts.length === 0) {
    return {
      artifact_analysis: [],
      artifact_summary: '未提供最终作品或提交内容。',
    };
  }

  const analyses: ArtifactAnalysis[] = [];

  for (const artifactEntry of artifacts) {
    const fileUrl = typeof artifactEntry === 'string' ? artifactEntry.trim() : '';

    if (!fileUrl) {
      analyses.push({
        file_url: fileUrl || '未提供链接',
        summary: '未提供可供解析的文件链接。',
        errors: ['缺少 file_url'],
      });
      continue;
    }

    if (!mineru_base_url) {
      analyses.push({
        file_url: fileUrl,
        summary: '文档拆解服务未配置，无法解析作品内容。',
        errors: ['缺少 MINERU_BASE_URL 配置'],
      });
      continue;
    }

    const errors: string[] = [];
    let mineru_payload_length = 0;
    let summary = '';

    try {
      const downloadedFile = await downloadArtifactFile(fileUrl);
      const mineruRawPayload = await callMineruWithImages(downloadedFile);
      summary = buildMineruSummary(mineruRawPayload);
    } catch (error) {
      errors.push(
        error instanceof Error
          ? `调用文档拆解服务失败: ${error.message}`
          : '调用文档拆解服务失败且未提供错误信息。',
      );
    }

    if (!summary) {
      summary = '文档拆解服务未返回摘要，可手动检查原始内容。';
    }

    analyses.push({
      file_url: fileUrl,
      mineru_payload_length: mineru_payload_length || undefined,
      summary,
      errors: errors.length > 0 ? errors : undefined,
    });
  }

  const artifact_summary = buildArtifactSummary(analyses);
  return {
    artifact_analysis: analyses,
    artifact_summary,
  };
}

async function assessProcess(state: typeof chainState.State) {
  const model = new ChatOpenAI({
    apiKey: openai_api_key,
    modelName: openai_chat_model,
    streaming: false,
  });

  const structuredLlm = model.withStructuredOutput(processAssessmentSchema);

  const process_assessment = await structuredLlm.invoke([
    {
      role: 'system',
      content: `你是一位具备教育测评与学习科学背景的教学诊断专家。请根据提供的学习计划、答题表现与互动记录，输出过程考查（process_assessment）结论，并严格遵循给定的 JSON schema。使用时先阅读摘要把握全局指标，再结合 JSON 证据补充细节。需要：
1. 针对知识体系中每个节点梳理掌握度、优势、问题、改进行动与证据；
2. 对掌握度、稳定性、迁移度三大维度给出等级判定、解释及支撑证据（如有波动或迁移不足需列出）；
3. 明确下一步的重点与练习建议（面向学生，避免“监控指标”“常规动作”等表述）；
4. 给出沟通要点或风险提示（聚焦可执行的短期行动，不涉及“长期监测”“研究跟进”）；
5. 结合学习投入、完成度、互动与反思等证据，按照提供的评分等级表计算过程考查的 total_score（0-100）与 grade（A-E），并说明评分依据与关键证据。
请只基于提供的数据，不得臆造。
注意：避免使用“子节点”“母节点”等术语；如需表达层级关系，请使用“知识点”“主题”“分项”等中性表述。`,
    },
    {
      role: 'human',
      content: `学习主题或课程: ${state.theme || '未提供'}`,
    },
    {
      role: 'human',
      content: `学生基础信息:\n${formatStudentInfo(state.student_info)}`,
    },
    {
      role: 'human',
      content: `学习计划与作答摘要（仅呈现全局指标与趋势，供你把握整体水平）:\n${state.plan_summary || '无'}`,
    },
    {
      role: 'human',
      content: `过程考查评分等级说明:
| 等级 | 分数区间 | 特征描述 |
| --- | --- | --- |
| A | 90-100 | 全程积极投入，学习频率高；能自主查找资料，知识掌握稳步提升；与同伴交流频繁，反思深刻、具有独立见解。 |
| B | 80-89 | 学习态度认真，完成任务及时；掌握水平有明显提升；能主动参与讨论并做出反思。 |
| C | 70-79 | 学习投入较稳定，能按要求完成任务；有一定提升但不显著；反思简单、合作有限。 |
| D | 60-69 | 学习积极性不足；任务常延迟或流于形式；进步缓慢或波动大；反思浅显。 |
| E | <60 | 学习参与度低，任务缺失严重；无明显成长；缺乏合作与反思。 |
请据此确定 total_score 与 grade，并在 interpretation 中写清判定理由。`,
    },
  ]);

  return { process_assessment };
}

async function assessPerformance(state: typeof chainState.State) {
  const openaiTimeoutMs = Number(process.env.OPENAI_TIMEOUT_MS ?? '0');

  const model = new ChatOpenAI({
    apiKey: openai_api_key,
    modelName: openai_chat_model,
    streaming: false,
    // 保护性超时（仅当 >0 时生效）
    timeout: openaiTimeoutMs > 0 ? openaiTimeoutMs : undefined,
  });

  const structuredLlm = model.withStructuredOutput(performanceAssessmentSchema);
  const artifactSummarySafe = truncateText(state.artifact_summary || '', TEXT_SUMMARY_MAX_LENGTH);

  try {
    const performance_assessment = await structuredLlm.invoke([
      {
        role: 'system',
        content: `你是一位以证据为中心的表现性评价（performance-based assessment）专家，擅长依据作品解析数据对学生成果进行可追溯、可解释、可量化的评价。你将接收一份作品的结构化解析数据（来自图像解析、OCR、文本抽取、内容识别等），并必须基于这些数据给出最终评价。你的任务是输出一个符合指定 JSON Schema 的 performance_assessment 对象，用于课程最终成绩评定。
要求：
1. 对每件作品在“内容质量、思维与创新、表达与呈现、规范与反思”四个维度分别给出 0-100 分的量化评分、等级（A-E）与评语；
2. 结合证据列出作品亮点、问题、改进行动与支撑证据；
3. 给出整体总结和面向学生的短期改进行动建议（避免“长期监测”“研究跟进”等表述）；
4. 严格依据数据，不得编造；
5. 表达与呈现只关注“逻辑是否被完整、清晰、可检验地呈现出来”。关注的是：读者能否从图表与结构中还原研究逻辑；关键内容是否齐全且可追踪；信息是否缺失、跳步或断裂。
 - 识别噪声规则（必须遵守）: 由于数据来自图像解析与OCR：拼写错误、错别字、断词、乱码、标点错位→ 不得作为“表达能力”“规范性”或“专业性”扣分依据。
6. 依据评分等级表计算最终表现的 overall_score（total_score、grade、interpretation），明确综合得分与判定理由。分数与等级必须严格一致，不得跨区间。
评分需参考以下等级说明：
| 等级 | 分数区间 | 特征描述 |
| --- | --- | --- |
| A | 90-100 | 逻辑严密、内容深入、创新显著、表达专业，具研究价值 |
| B | 80-89 | 内容扎实、思路清晰、略有创新、表达规范 |
| C | 70-79 | 内容基本正确、表达一般、创新性不足 |
| D | 60-69 | 存在明显逻辑或内容缺陷，表达欠清晰 |
| E | <60 | 内容错误多、结构混乱或可疑抄袭 |`,
      },
      {
        role: 'human',
        content: `学习主题或课程: ${state.theme || '未提供'}`,
      },
      {
        role: 'human',
        content: `学生基础信息:\n${formatStudentInfo(state.student_info)}`,
      },
      {
        role: 'human',
        content: `最终作品解析摘要（全局评分与总体结论，帮助你抓住宏观表现）:\n${artifactSummarySafe || '无'}`,
      },
    ]);

    return { performance_assessment };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    if (/aborted/i.test(message) || (error as any)?.name === 'AbortError') {
      throw new Error(
        `assessPerformance 被中断（aborted）。常见原因：\n- OpenAI 请求超时或被取消（可设置环境变量 OPENAI_TIMEOUT_MS 增大超时）；\n- 上下文过长导致请求过大（已做保护性截断，仍建议减少作品解析长度/数量）；\n- 短时网络中断或对端关闭连接。\n原始错误：${message}`,
      );
    }
    throw new Error(`assessPerformance 调用失败：${message}`);
  }
}

async function generatePortrait(state: typeof chainState.State) {
  const process = state.process_assessment;
  const performance = state.performance_assessment;

  if (!process || !performance) {
    return {};
  }

  const model = new ChatOpenAI({
    apiKey: openai_api_key,
    modelName: openai_chat_model,
    streaming: false,
  });

  const structuredOverview = model.withStructuredOutput(portraitOverviewSchema);
  const overviewInputPayload = JSON.stringify(
    {
      theme: state.theme || '未提供',
      student_profile: formatStudentInfo(state.student_info),
      process_assessment: process,
      performance_assessment: performance,
    },
    null,
    0,
  );

  const overviewResult = await structuredOverview.invoke([
    {
      role: 'system',
      content: `你是一位教学诊断专家。请基于提供的过程考查与最终表现结论，提炼一句概括学生当前水平、一句指出关键优势、一句强调优先改进方向，共 2-3 句话，严格返回 JSON。不得引入原数据之外的假设。`,
    },
    {
      role: 'human',
      content: overviewInputPayload,
    },
  ]);

  const portrait: StudentPortrait = {
    overview: overviewResult.overview,
    process_assessment: process,
    performance_assessment: performance,
  };

  return { portrait };
}

async function renderMarkdown(state: typeof chainState.State) {
  const portrait = state.portrait;

  if (!portrait) {
    return { portrait_markdown: '# 学生学习画像\n\n暂无画像结果。' };
  }

  const model = new ChatOpenAI({
    apiKey: openai_api_key,
    modelName: openai_chat_model_mini,
    streaming: false,
  });

  const structuredLlm = model.withStructuredOutput(portraitMarkdownSchema);
  const markdownInputPayload = JSON.stringify(
    {
      theme: state.theme || '未提供',
      student_profile: formatStudentInfo(state.student_info),
      overview: portrait.overview,
      process_assessment: portrait.process_assessment,
      performance_assessment: portrait.performance_assessment,
    },
    null,
    0,
  );

  const result = await structuredLlm.invoke([
    {
      role: 'system',
   content: `你是一位教学诊断专家，需要把既定的学生画像内容排版为 Markdown。请严格基于输入 JSON，保持事实一致。Markdown 必须包含三个部分：
    (1) “总体概览”：请基于提供的过程考查与最终表现结论，提炼一句概括学生当前水平、一句指出关键优势、一句强调优先改进方向，共 2-3 句话，并用自然语言给出两项评分及等级。例如：“过程性评分：85 分（等级 B）；表现性评分：88 分（等级 B）。” 不要在文本中出现任何字段名或路径（如 process_assessment.score.total_score、performance_assessment.overall_score 等），不要出现“=”“:”这类程序化标注。若某项数据缺失，仅写“未提供”。
    (2)“成绩与解读”：围绕过程考查与最终表现的关键结论/问题方向进行分点说明，覆盖掌握度、稳定性、迁移度等主要维度以及最终表现亮点或风险。不要描述“依据/证据来源/数据来源/字段名/schema”等元信息，只呈现客观结论与必要的事实描述。
    (3)“下一步建议”：整合 process_assessment.next_steps（priorities、recommended_practices）与 process_assessment.communication_notes，去重后按逻辑分组列出，聚焦学生可立即执行的有限任务与学习策略；禁止出现任何括注或方法性说明（如“按紧急与带动效应排序”“建议按……进行”），输出具体建议与要点。若信息缺失，仅写“未提供”。
其他约束：
 - 输出必须为有效的 Markdown 格式，包含标题、列表等结构化元素，便于阅读与理解；
 - 严禁输出任何内部变量名、字段路径或 schema 名称；
 - 不要使用“依据来源”“证据来源”“输入数据未提供XXX”之类表述；缺失项统一写“未提供”；
 - 禁止新增或删除事实，仅可为可读性进行轻微润色（包括去除括号内的排序/方法说明等元语句）；
 - 语气客观、中性，允许在事实支撑下作简短肯定性表述；读者为学生，避免“监控指标与常规动作”“长期监测与研究跟进”等表述；
 - 避免使用“子节点”“母节点”等术语，如需表达层级关系，用“知识点”“主题”“分项”等中性表述。`,
    },
    { role: 'human', content: markdownInputPayload },
  ]);

  await uploadPortraitMarkdownToS3(result.markdown, state.student_info);

  return {
    portrait_markdown: result.markdown,
  };
}

const workflow = new StateGraph(chainState)
  .addNode('summarizePlanDetail', summarizePlanDetail)
  .addNode('analyzeArtifacts', analyzeArtifacts)
  .addNode('assessProcess', assessProcess)
  .addNode('assessPerformance', assessPerformance)
  .addNode('generatePortrait', generatePortrait)
  .addNode('renderMarkdown', renderMarkdown)
  .addEdge('__start__', 'summarizePlanDetail')
  .addEdge('__start__', 'analyzeArtifacts')
  .addEdge('summarizePlanDetail', 'assessProcess')
  .addEdge('analyzeArtifacts', 'assessPerformance')
  .addEdge('assessProcess', 'generatePortrait')
  .addEdge('assessPerformance', 'generatePortrait')
  .addEdge('generatePortrait', 'renderMarkdown')
  .addEdge('renderMarkdown', '__end__');

export const graph = workflow.compile({
  // if you want to update the state before calling the tools
  // interruptBefore: [],
});
