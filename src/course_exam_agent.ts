import { Annotation, Send, StateGraph } from '@langchain/langgraph';
import { ChatOpenAI } from '@langchain/openai';
import { z } from 'zod';
import './instrumentation.js';

const openai_chat_model_mini = process.env.OPENAI_CHAT_MODEL_MINI ?? '';
const openai_api_key = process.env.OPENAI_API_KEY ?? '';

type userElement = {
  grade?: string;
  major_background?: string;
  grasp_level?: string;
  major?: string;
  history?: string[];
};

type QuestionType = 'SingleChoice' | 'MultipleChoices' | 'ShortAnswer';

const examPlanItemSchema = z.object({
  mergedLabel: z.string(),
  knowledgePoints: z.array(z.string()).min(1),
  questionType: z.enum(['SingleChoice', 'MultipleChoices', 'ShortAnswer']),
  questionCount: z.number().min(1).max(3),
  difficulty: z.number().min(3).max(5),
  focus: z.string(),
  rationale: z.string(),
  expectedSkills: z.array(z.string()).min(1),
  answerExpectations: z.string(),
});

type ExamPlanItem = z.infer<typeof examPlanItemSchema>;

const examPlanSchema = z.object({
  strategy: z.string(),
  blueprint: z.array(examPlanItemSchema).min(3),
});

type ExamPlan = z.infer<typeof examPlanSchema>;

const QUESTION_TYPE_LABEL: Record<QuestionType, string> = {
  SingleChoice: '单选题',
  MultipleChoices: '多选题',
  ShortAnswer: '简答题',
};

const QUESTION_TYPE_PROBLEM_TYPE: Record<QuestionType, number> = {
  SingleChoice: 1,
  MultipleChoices: 2,
  ShortAnswer: 5,
};

type RawKnowledgeNode = {
  id?: unknown;
  name?: unknown;
  title?: unknown;
  label?: unknown;
  children?: RawKnowledgeNode[];
  [key: string]: unknown;
};

type FlattenedKnowledge = {
  path: string[];
  title: string;
  description?: string;
  sampleQuestions: string[];
};

type KnowledgeParseResult = {
  outline: string;
  leafHighlights: string;
  detailMap: Record<string, string>;
  condensedRaw: string;
};

function condenseText(text: string, maxLength = 220): string {
  const cleaned = text.replace(/\s+/g, ' ').trim();
  if (cleaned.length <= maxLength) {
    return cleaned;
  }
  return `${cleaned.slice(0, maxLength - 1)}…`;
}

function safeToString(value: unknown): string {
  if (value === null || value === undefined) {
    return '';
  }
  if (typeof value === 'string') {
    return value;
  }
  if (typeof value === 'number' || typeof value === 'boolean') {
    return String(value);
  }
  try {
    return JSON.stringify(value);
  } catch (error) {
    return '';
  }
}

function summarizeKnowledgeContent(raw: string | undefined): KnowledgeParseResult {
  if (!raw) {
    return { outline: '', leafHighlights: '', detailMap: {}, condensedRaw: '' };
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch (error) {
    const condensed = condenseText(raw, 1800);
    return {
      outline: condensed,
      leafHighlights: condensed,
      detailMap: {},
      condensedRaw: condensed,
    };
  }

  const nodes: RawKnowledgeNode[] = Array.isArray(parsed)
    ? (parsed as RawKnowledgeNode[])
    : [parsed as RawKnowledgeNode];

  const leafNodes: FlattenedKnowledge[] = [];
  const branchInfo: Array<{ path: string[]; title: string; childCount: number }> = [];

  const traverse = (node: RawKnowledgeNode | undefined, path: string[]) => {
    if (!node || typeof node !== 'object') {
      return;
    }

    const titleCandidate = node.id ?? node.name ?? node.title ?? node.label ?? '';
    const currentTitle = safeToString(titleCandidate).trim() || `未命名节点${path.length + 1}`;
    const currentPath = [...path, currentTitle];

    const children = Array.isArray(node.children) ? node.children : [];
    const descriptionRaw = node['知识点描述'] ?? node['描述'] ?? node['description'];
    const description = condenseText(safeToString(descriptionRaw), 240);

    const examQuestionKeys = Object.keys(node).filter((key) => /^考试题目/.test(key));
    const sampleQuestions = examQuestionKeys
      .map((key) => {
        const text = safeToString(node[key]);
        const snippet = text.split('\n')[0] ?? text;
        return condenseText(snippet, 160);
      })
      .filter((item) => item.length > 0);

    if (children.length > 0) {
      branchInfo.push({ path: currentPath, title: currentTitle, childCount: children.length });
      children.forEach((child) => traverse(child, currentPath));
    }

    if (children.length === 0) {
      leafNodes.push({ path: currentPath, title: currentTitle, description, sampleQuestions });
    }
  };

  nodes.forEach((node) => traverse(node, []));

  const totalLeaves = leafNodes.length;
  const outlineParts: string[] = [];
  outlineParts.push(`知识结构共计 ${totalLeaves} 个末级知识点。`);

  const topLevelBranches = branchInfo.filter((branch) => branch.path.length === 2);
  topLevelBranches.forEach((branch, index) => {
    const leafUnderBranch = leafNodes.filter((leaf) =>
      branch.path.every((segment, idx) => leaf.path[idx] === segment),
    );
    const representatives = leafUnderBranch
      .slice(0, 3)
      .map((leaf) => leaf.title)
      .join('、');
    outlineParts.push(
      `${index + 1}. ${branch.path.join(' > ')}（叶子知识点 ${leafUnderBranch.length} 个）${
        representatives ? `，示例：${representatives}` : ''
      }`,
    );
  });

  const detailMap: Record<string, string> = {};
  const highlightEntries = leafNodes.slice(0, 60).map((leaf, idx) => {
    const detailParts: string[] = [];
    detailParts.push(`${idx + 1}. ${leaf.path.join(' > ')}`);
    if (leaf.description) {
      detailParts.push(`描述：${leaf.description}`);
    }
    if (leaf.sampleQuestions.length > 0) {
      detailParts.push(`典型考题：${leaf.sampleQuestions[0]}`);
    }

    const detailString = detailParts.join(' | ');

    const variants = new Set<string>();
    variants.add(leaf.title.trim());
    variants.add(leaf.path.join(' > ').trim());
    variants.add(leaf.path.slice(-2).join(' > ').trim());

    variants.forEach((key) => {
      const normalized = key.replace(/\s+/g, '');
      if (key && !detailMap[key]) {
        detailMap[key] = detailString;
      }
      if (normalized && !detailMap[normalized]) {
        detailMap[normalized] = detailString;
      }
    });

    return detailString;
  });

  branchInfo.forEach((branch) => {
    const detailString = `${branch.path.join(' > ')}（子节点 ${branch.childCount} 个）`;
    const key = branch.path.join(' > ').trim();
    if (key && !detailMap[key]) {
      detailMap[key] = detailString;
    }
    const normalized = key.replace(/\s+/g, '');
    if (normalized && !detailMap[normalized]) {
      detailMap[normalized] = detailString;
    }
  });

  const leafHighlights = highlightEntries.length
    ? `重点叶子知识点节选（共 ${totalLeaves} 个，展示 ${highlightEntries.length} 个）：\n${highlightEntries.join('\n')}`
    : `知识点原始数据：${condenseText(raw, 1800)}`;

  const condensedRaw = condenseText(raw, 1800);

  return {
    outline: outlineParts.join('\n'),
    leafHighlights,
    detailMap,
    condensedRaw,
  };
}

function lookupKnowledgeDetail(
  knowledgeMap: Record<string, string>,
  point: string,
): string | undefined {
  const trimmed = point.trim();
  if (!trimmed) {
    return undefined;
  }

  const normalized = trimmed.replace(/\s+/g, '');
  if (knowledgeMap[trimmed]) {
    return knowledgeMap[trimmed];
  }
  if (knowledgeMap[normalized]) {
    return knowledgeMap[normalized];
  }

  const exact = Object.entries(knowledgeMap).find(([key]) => key === trimmed);
  if (exact) {
    return exact[1];
  }

  const partial = Object.entries(knowledgeMap).find(
    ([key]) => key.includes(trimmed) || trimmed.includes(key),
  );
  if (partial) {
    return partial[1];
  }

  const lower = trimmed.toLowerCase();
  const caseInsensitive = Object.entries(knowledgeMap).find(([key]) =>
    key.toLowerCase().includes(lower),
  );
  if (caseInsensitive) {
    return caseInsensitive[1];
  }

  return undefined;
}

function formatStudentProfile(user: userElement): string {
  const profileSegments = [
    `年级：${user.grade ?? '未提供'}`,
    `专业：${user.major ?? '未提供'}`,
    `学术背景：${user.major_background ?? '未提供'}`,
    `掌握程度：${user.grasp_level ?? '未提供'}`,
  ];
  if (user.history?.length) {
    profileSegments.push(`历史薄弱点：${user.history.join('； ')}`);
  }
  return profileSegments.join('； ');
}

function buildQuestionInstruction(params: {
  course: string;
  planItem: ExamPlanItem;
  planStrategy: string;
  studentProfile: string;
  rawKnowledge?: string;
  order: number;
}): string {
  const { course, planItem, planStrategy, studentProfile, rawKnowledge, order } = params;
  const base = [
    `课程：${course}`,
    `学生画像：${studentProfile}`,
    `命题策略：${planStrategy || '关注学生薄弱点，保持较高区分度'}`,
    `重点考查组合：${planItem.mergedLabel}`,
    `覆盖知识点：${planItem.knowledgePoints.join('； ')}`,
    `目标能力：${planItem.expectedSkills.join('； ')}`,
    `题型：${QUESTION_TYPE_LABEL[planItem.questionType]}（第 ${order} 题）`,
    `难度要求：${planItem.difficulty}/5，务必具有挑战性，避免直接记忆型问答`,
    `命题聚焦：${planItem.focus}`,
    `答案要点：${planItem.answerExpectations}`,
    `命题理由：${planItem.rationale}`,
  ];

  if (rawKnowledge) {
    const trimmedKnowledge =
      rawKnowledge.length > 2000 ? `${rawKnowledge.slice(0, 2000)}...` : rawKnowledge;
    base.push(`相关知识点摘要：\n${trimmedKnowledge}`);
  }

  base.push(
    '整体要求：结合学生背景设计高阶思维问题，确保题干清晰并能拉开成绩分布，避免过于基础的直接记忆题。',
  );
  base.push(
    '公式格式要求：如需书写公式，请用单个 $ 包裹 LaTeX 形式（例如 $a^2 + b^2 = c^2$），不要使用双美元符号或其他标记。',
  );

  if (planItem.questionType === 'ShortAnswer') {
    base.push('该题需综合主观论述与必要计算步骤，明确给出评分要点。');
  }

  return base.join('\n');
}

const chainState = Annotation.Root({
  /**
   * 必填 - 课程名称或考试主题。
   */
  courseTitle: Annotation<string>(),
  /**
   * 必填 - 课程知识树的 JSON 字符串，结构如题目示例。
   */
  knowledgeTree: Annotation<string>(),
  /**
   * 可选 - 学生画像信息，缺省时使用兜底描述。
   */
  studentProfile: Annotation<userElement>(),
  /**
   * 自动生成 - 为命题蓝图准备的知识结构纲要。
   */
  knowledgeOutline: Annotation<string>({ reducer: (_, y) => y, default: () => '' }),
  /**
   * 自动生成 - 末级知识点精要（用于题目上下文）。
   */
  knowledgeQuestionDigest: Annotation<string>({ reducer: (_, y) => y, default: () => '' }),
  /**
   * 自动生成 - 知识点名称对应的详情映射。
   */
  knowledgeMap: Annotation<Record<string, string>>({ reducer: (_, y) => y, default: () => ({}) }),
  /**
   * 自动生成 - 命题策略描述。
   */
  planStrategy: Annotation<string>({ reducer: (_, y) => y, default: () => '' }),
  /**
   * 自动生成 - 命题蓝图。
   */
  examPlan: Annotation<ExamPlanItem[]>({ reducer: (_, y) => y, default: () => [] }),
  /**
   * 内部使用 - 当前命题蓝图条目。
   */
  planItem: Annotation<ExamPlanItem | undefined>({
    reducer: (_, y) => y,
    default: () => undefined,
  }),
  /**
   * 内部使用 - 传递给题目生成模型的具体指令。
   */
  instructions: Annotation<string>(),
  /**
   * 自动聚合 - 模型生成的题目列表。
   */
  questions: Annotation<unknown[]>({ reducer: (x, y) => (x || []).concat(y), default: () => [] }),
  /**
   * 输出 - 对外暴露的结果数组。
   */
  output: Annotation<unknown[]>(),
});

async function prepareKnowledgeContext(state: typeof chainState.State) {
  const rawDescriptions =
    typeof state.knowledgeTree === 'string'
      ? state.knowledgeTree
      : safeToString(state.knowledgeTree);
  const summary = summarizeKnowledgeContent(rawDescriptions);
  return {
    knowledgeOutline: summary.outline,
    knowledgeQuestionDigest: summary.leafHighlights,
    knowledgeMap: summary.detailMap,
    knowledgeTree: summary.condensedRaw,
  };
}

async function prepareExamPlan(state: typeof chainState.State) {
  const studentProfile = formatStudentProfile(state.studentProfile ?? {});
  const model = new ChatOpenAI({
    apiKey: openai_api_key,
    modelName: openai_chat_model_mini,
    streaming: false,
  });

  const structuredLlm = model.withStructuredOutput(examPlanSchema);

  const response = await structuredLlm.invoke([
    {
      role: 'system',
      content:
        '负责根据学生画像和课程考核点设计高难度试题。请先对考点进行合并与归类，再匹配合适题型。题型只能是 SingleChoice、MultipleChoices、ShortAnswer 三种字符串之一，并确保难度系数在 3-5 范围。请覆盖全部三种题型且每种至少安排一题。短答题需要包含主观论述与必要计算，可适当分配多题。如果考点超过 20 个，请先分组再命题。',
    },
    {
      role: 'human',
      content: `课程名称：${state.courseTitle}
学生画像：${studentProfile}
知识结构总览：
${state.knowledgeOutline || state.knowledgeTree}

叶子知识点节选：
${state.knowledgeQuestionDigest || state.knowledgeTree}`,
    },
  ]);

  const plan: ExamPlan = {
    strategy: response.strategy,
    blueprint: response.blueprint,
  };

  return { examPlan: plan.blueprint, planStrategy: plan.strategy };
}

function routeExamPlan(state: typeof chainState.State): Send[] {
  if (!state.examPlan || state.examPlan.length === 0) {
    return [];
  }

  const studentProfile = formatStudentProfile(state.studentProfile ?? {});
  let questionOrder = 0;

  return state.examPlan.flatMap((planItem) => {
    const planCount = Math.max(1, Math.min(3, planItem.questionCount));
    const sends: Send[] = [];
    const knowledgeDetails: string[] = [];
    const detailSet = new Set<string>();

    planItem.knowledgePoints.forEach((point) => {
      const detail = lookupKnowledgeDetail(state.knowledgeMap || {}, point);
      if (detail && !detailSet.has(detail)) {
        detailSet.add(detail);
        knowledgeDetails.push(detail);
      }
    });

    const knowledgeContext = knowledgeDetails.length
      ? `关联知识点精要：\n${knowledgeDetails
          .slice(0, 8)
          .map((detail, idx) => `${idx + 1}. ${detail}`)
          .join('\n')}`
      : state.knowledgeQuestionDigest || state.knowledgeTree;

    for (let i = 0; i < planCount; i += 1) {
      questionOrder += 1;
      sends.push(
        new Send(planItem.questionType, {
          planItem,
          planStrategy: state.planStrategy,
          instructions: buildQuestionInstruction({
            course: state.courseTitle ?? '',
            planItem,
            planStrategy: state.planStrategy,
            studentProfile,
            rawKnowledge: knowledgeContext,
            order: questionOrder,
          }),
        }),
      );
    }
    return sends;
  });
}

async function SingleChoice(state: typeof chainState.State) {
  const planItem = state.planItem;
  if (!planItem) {
    throw new Error('SingleChoice node requires planItem in state');
  }

  const singlechoiceSchema = z.object({
    Body: z.string().describe('Question body'),
    Options: z.array(
      z.object({
        key: z.enum(['A', 'B', 'C', 'D']),
        value: z.string().describe('option'),
      }),
    ),
    Answer: z.array(z.enum(['A', 'B', 'C', 'D'])).describe('One correct answer to this question'),
    difficulty: z.number().describe('difficulty level of the question, 1 means easy, 5 means hard'),
    Remark: z.string().describe('explanation of the answer'),
  });

  const model = new ChatOpenAI({
    apiKey: openai_api_key,
    modelName: openai_chat_model_mini,
    streaming: false,
  });

  const structuredLlm = model.withStructuredOutput(singlechoiceSchema);

  const response = await structuredLlm.invoke([
    {
      role: 'system',
      content:
        '你是一名经验丰富的大学考试命题专家。当前需要输出高难度单选题，确保选项具有迷惑性并考查高阶思维能力。',
    },
    {
      role: 'human',
      content: `${state.instructions ?? ''}

产出要求：
- 语言：中文。
- 题干要结合情境，避免直接记忆题。
- 选项需互斥且具有迷惑性，确保只有一个正确答案。
- 难度系数不低于 ${planItem.difficulty}。
- 在解析中，必须解释正确选项的原因并逐一指出干扰项的误区。
- 公式格式要求：如需书写公式，请用单个 $ 包裹 LaTeX 形式（例如 $a^2 + b^2 = c^2$），不要使用双美元符号或其他标记。
- 前端展示转义规则（仅做必要转义，不改内容）：将所有表示“金额/成本/单位成本”等且不在数学公式中的美元符号 "$" 输出为 "\\$"；不要改动任何数学公式的 "$...$" 或 "$$...$$" 结构（含定界符）；除上述转义外，不修改数字、单位、百分比、变量名或句子结构；不新增、不删除任何信息；输出最终整理后的完整文本。
`,
    },
  ]);

  const enrichedResponse = {
    ...response,
    difficulty: Math.max(planItem.difficulty, response.difficulty),
    Type: 'SingleChoice',
    TypeText: QUESTION_TYPE_LABEL.SingleChoice,
    ProblemType: QUESTION_TYPE_PROBLEM_TYPE.SingleChoice,
    // Blueprint: planItem,
  };

  return { questions: enrichedResponse };
}

async function MultipleChoices(state: typeof chainState.State) {
  const planItem = state.planItem;
  if (!planItem) {
    throw new Error('MultipleChoices node requires planItem in state');
  }

  const multiplechoicesSchema = z.object({
    Body: z.string().describe('Question body'),
    Options: z.array(
      z.object({
        key: z.enum(['A', 'B', 'C', 'D']),
        value: z.string().describe('option'),
      }),
    ),
    Answer: z
      .array(z.enum(['A', 'B', 'C', 'D']))
      .describe('A set of correct answers to this question'),
    difficulty: z.number().describe('difficulty level of the question, 1 means easy, 5 means hard'),
    Remark: z.string().describe('explanation of the answer'),
  });

  const model = new ChatOpenAI({
    apiKey: openai_api_key,
    modelName: openai_chat_model_mini,
    streaming: false,
  });

  const structuredLlm = model.withStructuredOutput(multiplechoicesSchema);

  const response = await structuredLlm.invoke([
    {
      role: 'system',
      content:
        '你是一名大学高阶能力评估专家。请根据给定蓝图设计高难度多选题，至少包含两个正确选项，干扰项需源自常见误区。',
    },
    {
      role: 'human',
      content: `${state.instructions ?? ''}

产出要求：
- 语言：中文。
- 题干应强调综合分析、比较或推理，避免纯记忆描述。
- 设置至少两个正确答案，其余选项需要针对性干扰，体现常见混淆点。
- 难度系数不低于 ${planItem.difficulty}。
- 解析中先整体说明，再分别阐释每个正确选项和错误选项。
- 公式格式要求：如需书写公式，请用单个 $ 包裹 LaTeX 形式（例如 $a^2 + b^2 = c^2$），不要使用双美元符号或其他标记。
- 前端展示转义规则（仅做必要转义，不改内容）：将所有表示“金额/成本/单位成本”等且不在数学公式中的美元符号 "$" 输出为 "\\$"；不要改动任何数学公式的 "$...$" 或 "$$...$$" 结构（含定界符）；除上述转义外，不修改数字、单位、百分比、变量名或句子结构；不新增、不删除任何信息；输出最终整理后的完整文本。`,
    },
  ]);

  const enrichedResponse = {
    ...response,
    difficulty: Math.max(planItem.difficulty, response.difficulty),
    Type: 'MultipleChoice',
    TypeText: QUESTION_TYPE_LABEL.MultipleChoices,
    ProblemType: QUESTION_TYPE_PROBLEM_TYPE.MultipleChoices,
    // Blueprint: planItem,
  };

  return { questions: enrichedResponse };
}

async function ShortAnswer(state: typeof chainState.State) {
  const planItem = state.planItem;
  if (!planItem) {
    throw new Error('ShortAnswer node requires planItem in state');
  }

  const shortanswerSchema = z.object({
    Body: z.string().describe('Question body'),
    difficulty: z.number().describe('difficulty level of the question, 1 means easy, 5 means hard'),
    Remark: z
      .string()
      .describe('explanation of the answer including key points needed for grading'),
  });

  const model = new ChatOpenAI({
    apiKey: openai_api_key,
    modelName: openai_chat_model_mini,
    streaming: false,
  });

  const structuredLlm = model.withStructuredOutput(shortanswerSchema);

  const response = await structuredLlm.invoke([
    {
      role: 'system',
      content:
        '请根据命题蓝图生成需要主观论述与计算分析的高难度简答题，并提供详尽评分要点。',
    },
    {
      role: 'human',
      content: `${state.instructions ?? ''}

产出要求：
- 语言：中文。
- 题干需包含真实案例或复杂情境，引导学生分析并进行必要的计算或推导。
- 难度系数不低于 ${planItem.difficulty}。
- 评分要点需拆分为可量化的子项，包含计算步骤、关键结论与论证逻辑。
- 公式格式要求：如需书写公式，请用单个 $ 包裹 LaTeX 形式（例如 $a^2 + b^2 = c^2$），不要使用双美元符号或其他标记。
- 前端展示转义规则（仅做必要转义，不改内容）：将所有表示“金额/成本/单位成本”等且不在数学公式中的美元符号 "$" 输出为 "\\$"；不要改动任何数学公式的 "$...$" 或 "$$...$$" 结构（含定界符）；除上述转义外，不修改数字、单位、百分比、变量名或句子结构；不新增、不删除任何信息；输出最终整理后的完整文本。`,
    },
  ]);

  const enrichedResponse = {
    ...response,
    difficulty: Math.max(planItem.difficulty, response.difficulty),
    Type: 'ShortAnswer',
    TypeText: '简答题',
    ProblemType: QUESTION_TYPE_PROBLEM_TYPE.ShortAnswer,
    // Blueprint: planItem,
  };

  return { questions: enrichedResponse };
}

async function outputQuestions(state: typeof chainState.State) {
  return { output: state.questions };
}

const workflow = new StateGraph(chainState)
  .addNode('prepareKnowledge', prepareKnowledgeContext)
  .addNode('prepareExamPlan', prepareExamPlan)
  .addNode('SingleChoice', SingleChoice)
  .addNode('MultipleChoices', MultipleChoices)
  .addNode('ShortAnswer', ShortAnswer)
  .addNode('outputQuestions', outputQuestions)
  .addEdge('__start__', 'prepareKnowledge')
  .addEdge('prepareKnowledge', 'prepareExamPlan')
  .addConditionalEdges('prepareExamPlan', routeExamPlan, [
    'SingleChoice',
    'MultipleChoices',
    'ShortAnswer',
  ])
  .addEdge('SingleChoice', 'outputQuestions')
  .addEdge('MultipleChoices', 'outputQuestions')
  .addEdge('ShortAnswer', 'outputQuestions')
  .addEdge('outputQuestions', '__end__');

export const graph = workflow.compile({
  // if you want to update the state before calling the tools
  // interruptBefore: [],
});
