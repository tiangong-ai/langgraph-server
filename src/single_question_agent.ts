import { Annotation, StateGraph } from '@langchain/langgraph';
import { ChatOpenAI } from '@langchain/openai';
import { z } from 'zod';
import './instrumentation.js';

const openai_chat_model_mini = process.env.OPENAI_CHAT_MODEL_MINI ?? '';
const openai_chat_model = process.env.OPENAI_CHAT_MODEL ?? '';
const openai_api_key = process.env.OPENAI_API_KEY ?? '';

type questionElement = {
  question: string;
  complete: string;
};

const optionSchema = z.object({
  key: z.enum(['A', 'B', 'C', 'D']),
  value: z.string().describe('option'),
});

const optionsArraySchema = z
  .array(optionSchema)
  .length(4)
  .superRefine((options, ctx) => {
    const keySet = new Set<string>();
    const valueMap = new Map<string, number>();

    options.forEach((option, index) => {
      if (keySet.has(option.key)) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: 'Each option key must be unique.',
          path: [index, 'key'],
        });
      }
      keySet.add(option.key);

      const normalizedValue = option.value.trim();
      if (valueMap.has(normalizedValue)) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: 'Each option text must be unique.',
          path: [index, 'value'],
        });
      } else {
        valueMap.set(normalizedValue, index);
      }
    });
  });

const chainState = Annotation.Root({
  knowledge_point: Annotation<string>(),
  knowledge_descriptions: Annotation<string>(),
  question_history: Annotation<questionElement[]>(),
  question_type: Annotation<number>(),
  difficulty: Annotation<number>(),
  output: Annotation<string>(),
});

function routeQuestionType(state: typeof chainState.State) {
  if ((state.knowledge_descriptions?.length ?? 0) > 0 && state.question_type === 1) {
    return 'SingleChoice';
  }
  return 'MultipleChoices';
}

async function SingleChoice(state: typeof chainState.State) {
  const singlechoiceSchema = z.object({
    Body: z
      .string()
      .describe('Only the question body, dont include the options and also other information.'),
    Options: optionsArraySchema,
    Answer: z.array(z.enum(['A', 'B', 'C', 'D'])).describe('One correct answer to this question'),
    // difficulty: z.enum(['1', '2']).describe('difficulty level of the question, 1 means basic, 2 means hard'),
    Remark: z.string().describe('explanation of the answer'),
  });

  const model = new ChatOpenAI({
    apiKey: openai_api_key,
    model: openai_chat_model_mini,
    streaming: false,
  });

  const structuredLlm = model.withStructuredOutput(singlechoiceSchema);

  const response = await structuredLlm.invoke([
    {
      role: 'system',
      content: `According to the given knowledge descriptions, generate a Single-choice question related to '${state.knowledge_point ?? ''}' with the following guidelines:
- Language: Chinese.
- Question body: The question should focus on core concepts or key facts that require the user to recall and apply the information.
- Question options: Provide four options labeled A, B, C, and D, with one correct answer and three distractors. Distractors should be plausible but incorrect, requiring careful consideration to identify the correct answer.
- Answer: Indicate the correct answer by specifying the corresponding option (A, B, C, or D).
- Explanation: Provide a clear and concise explanation of the correct answer, helping students understand why the answer is correct and why the other options are incorrect.
- Avoid extreme terms: Do not include extreme terms like “always” or “never” in the options, as they are easily ruled out by students.
- Question clarity: The question should be clear and concise, ensuring students can easily understand what is being asked.
- Knowledge alignment: Do not extend beyond the provided knowledge, strictly base questions only on the information given in the knowledge descriptions, without adding external knowledge or concepts.
- Formula formatting: If a formula is needed, wrap LaTeX content with a single $ (e.g., $a^2 + b^2 = c^2$); do not use double dollars or other delimiters.
- Question uniqueness: Do not repeat questions that have been asked before. Review the question history and ensure the new question is different from previous ones.`,
    },
    {
      role: 'human',
      content: `Knowledge descriptions: ${state.knowledge_descriptions ?? ''}`,
    },
    {
      role: 'human',
      content: `Question history: ${
        state.question_history?.map((q) => q.question).join('\n ') ?? ''
      }`,
    },
  ]);

  const enrichedResponse = {
    ...response,
    Type: 'SingleChoice',
    TypeText: '单选题',
    Difficulty: state.difficulty,
    ProblemType: 1,
  };

  return { output: enrichedResponse };
}

async function MultipleChoices(state: typeof chainState.State) {
  const multiplechoicesSchema = z.object({
    Body: z
      .string()
      .describe('Only the question body, dont include the options and also other information'),
    Options: optionsArraySchema,
    Answer: z
      .array(z.enum(['A', 'B', 'C', 'D']))
      .describe('A set of correct answers to this question'),
    Remark: z.string().describe('explanation of the answer'),
  });

  const model = new ChatOpenAI({
    apiKey: openai_api_key,
    model: openai_chat_model,
    streaming: false,
  });

  const structuredLlm = model.withStructuredOutput(multiplechoicesSchema);

  const response = await structuredLlm.invoke([
    {
      role: 'system',
      content: `Generate a Multiple-choice question (Focus on higher-order thinking skills rather than basic comprehension of knowledge descriptions) related to '${state.knowledge_point ?? ''}' with the following guidelines:
- Language: Must be in Chinese.
- Question body: The question should focus on core concepts or key facts that require the user to recall and apply the information.
- Question options: Provide four options labeled A, B, C, and D, with at least two correct answer and other distractors. Ensure distractors are sophisticated distractors that: draw from related but distinct concepts, represent common student errors in reasoning, include elements that might be true in different contexts, and require domain knowledge to identify as incorrect.
- Answer: Indicate the correct answer by specifying the corresponding option (A, B, C, or D).
- Explanation: Provide a clear and concise explanation of these correct answer, helping students understand why the answer is correct and why the other options are incorrect.
- Avoid extreme terms: Do not include extreme terms like “always”, "only", "just" or “never” in the options, as they are easily ruled out by students.
- Question uniqueness: Do not repeat questions that have been asked before. Review the question history and ensure the new question is different from previous ones.
- Question clarity: The question should be clear and concise, ensuring students can easily understand what is being asked.
- Formula formatting: If a formula is needed, wrap LaTeX content with a single $ (e.g., $a^2 + b^2 = c^2$); do not use double dollars or other delimiters.
- Question complexity: Design questions that:
  * Involve multi-step reasoning
  * Require integration of different knowledge areas
  * Test understanding of cause-and-effect relationships
  * Challenge common misconceptions
- Knowledge extension: While using the provided knowledge descriptions as foundation, extend to:
  * Real-world applications and case studies
  * Cross-disciplinary connections
  * Contemporary developments and trends
  * Practical implications and problem-solving scenarios.
`,
    },
    {
      role: 'human',
      content: `Knowledge descriptions: ${state.knowledge_descriptions ?? ''}`,
    },
    {
      role: 'human',
      content: `Question history: ${
        state.question_history?.map((q) => q.question).join('\n ') ?? ''
      }`,
    },
  ]);

  const enrichedResponse = {
    ...response,
    Type: 'MultipleChoice',
    TypeText: '多选题',
    Difficulty: state.difficulty,
    ProblemType: 2,
  };

  return { output: enrichedResponse };
}

const workflow = new StateGraph(chainState)
  .addNode('SingleChoice', SingleChoice)
  .addNode('MultipleChoices', MultipleChoices)
  .addConditionalEdges('__start__', routeQuestionType, ['SingleChoice', 'MultipleChoices'])
  .addEdge('SingleChoice', '__end__')
  .addEdge('MultipleChoices', '__end__');

export const graph = workflow.compile({
  // if you want to update the state before calling the tools
  // interruptBefore: [],
});
``;
