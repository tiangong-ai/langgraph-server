import { Annotation, StateGraph } from '@langchain/langgraph';
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

const singleChoiceSchema = z.object({
  Type: z.literal('SingleChoice'),
  TypeText: z.literal('单选题'),
  ProblemType: z.literal(1),
  Body: z.string().describe('题干内容（中文）。不要在此处包含选项标签或答案。'),
  Options: z
    .array(
      z.object({
        key: z.enum(['A', 'B', 'C', 'D']),
        value: z.string().describe('选项内容（中文）。'),
      }),
    )
    .min(2)
    .max(4)
    .superRefine((options, ctx) => {
      const allowedOrder = ['A', 'B', 'C', 'D'];
      const seenKeys = new Set<string>();
      options.forEach((option, index) => {
        if (seenKeys.has(option.key)) {
          ctx.addIssue({
            code: z.ZodIssueCode.custom,
            message: '选项键不能重复。',
            path: [index, 'key'],
          });
        }
        seenKeys.add(option.key);
      });
      if (options.length > 0) {
        if (options[0].key !== 'A') {
          ctx.addIssue({
            code: z.ZodIssueCode.custom,
            message: '第一项必须使用选项 A。',
            path: [0, 'key'],
          });
        }
        for (let i = 1; i < options.length; i += 1) {
          const expectedKey = allowedOrder[i];
          if (options[i].key !== expectedKey) {
            ctx.addIssue({
              code: z.ZodIssueCode.custom,
              message: '选项必须按照 A、B、C、D 的顺序连续排列，不得跳过。',
              path: [i, 'key'],
            });
          }
        }
      }
    })
    .describe('提供 2-4 个选项，键按从 A 开始连续排列。'),
  Answer: z
    .array(z.enum(['A', 'B', 'C', 'D']))
    .length(1)
    .describe('仅有 1 个正确答案，用其选项键表示。'),
  difficulty: z.number().describe('难度等级 1（易）至 5（难），与样题难度一致。'),
  Remark: z.string().describe('说明答案正确的原因，并引用相关知识内容。'),
});

const multipleChoicesSchema = z.object({
  Type: z.literal('MultipleChoice'),
  TypeText: z.literal('多选题'),
  ProblemType: z.literal(2),
  Body: z.string().describe('题干内容（中文）。不要包含选项标签或答案。'),
  Options: z
    .array(
      z.object({
        key: z.enum(['A', 'B', 'C', 'D']),
        value: z.string().describe('选项内容（中文）。'),
      }),
    )
    .length(4)
    .describe('提供 A-D 共 4 个选项。'),
  Answer: z
    .array(z.enum(['A', 'B', 'C', 'D']))
    .min(2)
    .describe('至少 2 个正确答案，用其选项键表示。'),
  difficulty: z.number().describe('难度等级 1（易）至 5（难），与样题难度一致。'),
  Remark: z.string().describe('解释每个选项正确或错误的原因。'),
});

const responseSchema = z.object({
  questions: z
    .array(z.union([singleChoiceSchema, multipleChoicesSchema]))
    .length(3)
    .describe('严格输出 3 道题。每题应根据对应样题信息选择匹配的客观题型架构。'),
});

type GeneratedQuestion = z.infer<typeof responseSchema>['questions'][number];

const chainState = Annotation.Root({
  knowledge_point_name: Annotation<string>(),
  knowledge_content: Annotation<string>(),
  sample_questions: Annotation<string[]>(),
  user_context: Annotation<userElement>(),
  output: Annotation<GeneratedQuestion[]>(),
});

async function GenerateQuizQuestions(state: typeof chainState.State) {
  const userContext = state.user_context ?? {};
  const today = new Date().toISOString().slice(0, 10);
  const seedComponents = [
    userContext.grade ?? '',
    userContext.major ?? '',
    today,
  ].filter((component) => component && component.length > 0);
  const variantSeed = seedComponents.length > 0 ? seedComponents.join('|') : today;

  const userContextEntries = [
    userContext.grade ? `年级线索：${userContext.grade}` : '',
    userContext.major ? `学科方向：${userContext.major}` : '',
    userContext.major_background ? `既有背景：${userContext.major_background}` : '',
    userContext.grasp_level ? `掌握程度：${userContext.grasp_level}` : '',
    (userContext.history?.length ?? 0) > 0
      ? `常见易错点提示：易混淆于${userContext.history?.join('； ')}`
      : '',
  ].filter((entry) => entry.length > 0);

  const userContextSummary = userContextEntries.length === 0 ? '无' : userContextEntries.join('； ');
  const userContextPrompt =
    userContextSummary === '无'
      ? '学生上下文：无。'
      : `学生上下文（仅用于把握命题侧重点和难度，题干中严禁出现与学生相关的描述或身份信息）：${userContextSummary}`;

  const model = new ChatOpenAI({
    apiKey: openai_api_key,
    model: openai_chat_model_mini,
    streaming: false,
  });

  const structuredLlm = model.withStructuredOutput(responseSchema);

  const response = await structuredLlm.invoke([
    {
      role: 'system',
      content: `你是一名教学设计专家，负责根据给定的样题生成高度相似且不重复的练习题。
请严格遵循以下要求：
- 语言：全程使用中文编写题干、选项、答案与解析。
- 前端展示转义规则（仅做必要转义，不改内容）：将所有表示“金额/成本/单位成本”等且不在数学公式中的美元符号 \`$\` 输出为 \`\\$\`；不要改动任何数学公式的 \`$...$\` 或 \`$$...$$\` 结构（含定界符）；除上述转义外，不修改数字、单位、百分比、变量名或句子结构；不新增、不删除任何信息；输出最终整理后的完整文本。
- 表述要求：题干及解析保持客观、中性，采用第三人称或情境化描述，避免使用“你”“您的”等第一、第二人称，并不得直接引用具体学生身份或点名；如需引用学生上下文，请以场景设定或第三方叙述方式呈现，仅可使用“某类”“面向”等泛化称谓，禁止出现“该学生”“该博士”等措辞。
- 题干要求：题干必须专注于问题本身，不得包含学生背景、教学提示或任何补充说明；上下文信息仅可用于调整题目设定思路而非直接写入题干。
- 题型规范：严格输出客观题，严禁生成主观题，仅允许以下结构：
  * 单选题：仅 1 个正确选项。
  * 多选题：存在多个正确选项。
- 公式格式：如需书写公式，请用单个 $ 包裹 LaTeX（例如 $a^2 + b^2 = c^2$），不要使用双美元符号或其他标记。
- 样题映射：逐条分析样题，并一一对应生成新题：
  * 总共输出 3 道题，顺序与输入样题一致。
  * 若样题为单选题或出现唯一正确要点的问答题，则输出单选题。
  * 若样题为多选题或需要多个关键要点，则输出多选题。
  * 若样题为判断题（对/错），请转换为单选题，并且必须严格满足：Options.length = 2；仅允许 A：正确、B：错误；严禁输出 C、D 或任何占位文本；答案只能是 A 或 B 且唯一正确。
  * 若样题为主观题，请先梳理其核心考点，再设计成选项清晰的客观题：考点只有一个正确判断时输出单选题，涉及多个同时正确要点时输出多选题。
- 知识对齐：题干、答案与解析必须与提供的知识内容一致，并明确指向给定的知识点。
- 相似不重复：保持与样题相同的题型与思维路径，但更换情境、角色、数据或措辞，避免与样题重复。
- 结构映射：保持与样题相同的逻辑结构、难度与认知要求，同时更换表述与背景以避免重复。
- 多样性：三道题应体现不同角度或应用场景。
- 难度：每题的难度值（1-5）应与其对应样题的隐含难度保持一致，不能过于简单。
- 解析：请在解析中明确将推理或关键点与知识内容对应。
- 变体种子（用于不同学生有差异）：${variantSeed}。`,
    },
    {
      role: 'human',
      content: `知识点名称: ${state.knowledge_point_name ?? ''}\n知识点内容: ${
        state.knowledge_content ?? ''
      }\n请基于该知识点生成三道新题目。`,
    },
    {
      role: 'human',
      content: userContextPrompt,
    },
    {
      role: 'human',
      content: `示例题目（逐条对应生成一题）：\n${(state.sample_questions ?? [])
        .map((question, index) => `${index + 1}. ${question}`)
        .join('\n')}`,
    },
  ]);

  return { output: response.questions };
}

const workflow = new StateGraph(chainState)
  .addNode('GenerateQuizQuestions', GenerateQuizQuestions)
  .addEdge('__start__', 'GenerateQuizQuestions')
  .addEdge('GenerateQuizQuestions', '__end__');

export const graph = workflow.compile({
  // if you want to update the state before calling the tools
  // interruptBefore: [],
});
