import { Neo4jGraph } from '@langchain/community/graphs/neo4j_graph';
import { Annotation, Send, StateGraph } from '@langchain/langgraph';
import { ChatOpenAI } from '@langchain/openai';
import { z } from 'zod';
import './instrumentation.js';

const openai_chat_model_mini = process.env.OPENAI_CHAT_MODEL_MINI ?? '';
const openai_api_key = process.env.OPENAI_API_KEY ?? '';
const question_types = ['SingeChoice', 'MultipleChoices', 'ShortAnswer'];

type userElement = {
  grade?: string;
  major_background?: string;
  grasp_level?: string;
  major?: string;
  history?: string[];
};

const chainState = Annotation.Root({
  input: Annotation<string>(),
  descriptions: Annotation<string>(),
  userData: Annotation<userElement>(),
  portraitData: Annotation<string>(),
  graphData: Annotation<string>(),
  numbers: Annotation<number>({ reducer: (_, y) => y, default: () => 1 }),
  instructions: Annotation<string>(),
  questions: Annotation<string[]>({ reducer: (x, y) => (x || []).concat(y), default: () => [] }),
  output: Annotation<string[]>(),
});

async function getGraph(state: typeof chainState.State) {
  const url = process.env.NEO4J_URI ?? '';
  const username = process.env.NEO4J_USER ?? '';
  const password = process.env.NEO4J_PASSWORD ?? '';
  const graph = await Neo4jGraph.initialize({ url, username, password });
  const graphData = await graph.query(
    `
    CALL db.index.fulltext.queryNodes("concept_fulltext_index", "${state.input}") YIELD node, score
    WITH node, score
    ORDER BY score DESC
    LIMIT 1
    MATCH p = (:Concept)-[r1:HAS_PART]->(n)-[*1..2]->()
    WHERE n.id = node.id
    RETURN p LIMIT 30
    `,
  );
  const graphData_json = JSON.stringify(graphData);
  const graphData_list: string[] = [];
  JSON.parse(graphData_json).map(
    (path: {
      p: Array<{
        CATEGORY?: string;
        id?: string;
      }>;
    }) => {
      const pathData_set = new Set<string>();
      path['p'].map((data) => {
        if (data.CATEGORY && data.id) {
          pathData_set.add(data.id);
        }
      });
      graphData_list.push(Array.from(pathData_set).join('->'));
    },
  );
  const graphData_string: string = graphData_list.join('\n');
  return { graphData: graphData_string };
}

async function routeSimpleQustions(state: typeof chainState.State): Promise<Send[]> {
  const SYSTEM_PROMPT = `Generate a very simple question based on the following topic —— ${state.input}, specifically targeting potential knowledge gaps identified in the student's historical interactions.
  Review their conversation history (${state.userData.history?.join('； ')}) to identify areas where they may have misconceptions or incomplete understanding.
  The question should help address these specific knowledge gaps while reinforcing core concepts.
  After the question, provide a clear and concise answer that explains the key point and helps clarify the concept, with special attention to the student's academic background (${state.userData.grade ?? '本科生'}, ${state.userData.major_background ?? '没有经验'}) and current grasp level (${state.userData.grasp_level ?? '不太熟练'}).
  Ouput these questions in ** Chinese **.
  `;

  return question_types.map((type) => {
    return new Send(type, {
      instructions: SYSTEM_PROMPT,
      descriptions: state.descriptions,
      graphData: state.graphData,
    });
  });
}

async function routeMediumQustions(state: typeof chainState.State): Promise<Send[]> {
  const SYSTEM_PROMPT = `Generate a medium-difficulty exam question that tests the understanding of key concepts in ${state.input}, specifically targeting potential knowledge gaps identified in the student's historical interactions. 
  Review their conversation history (${state.userData.history?.join('； ')}) to identify areas where they may have misconceptions or incomplete understanding.
  The question should require the respondent to demonstrate their comprehension of the material and apply their knowledge to a relevant scenario.
  Ensure the question is challenging enough to stimulate critical thinking and encourage a deeper understanding of the topic.
  After the question, provide a clear and concise answer that explains the key point and helps clarify the concept, with special attention to the student's academic background (${state.userData.grade ?? '硕士生'}, ${state.userData.major_background ?? '具有一定经验'}) and current grasp level (${state.userData.grasp_level ?? '基本掌握'}).
  Ouput these questions in ** Chinese **.`;
  return question_types.map((type) => {
    return new Send(type, {
      instructions: SYSTEM_PROMPT,
      descriptions: state.descriptions,
      graphData: state.graphData,
    });
  });
}

async function routeHardQustions(state: typeof chainState.State): Promise<Send[]> {
  const SYSTEM_PROMPT = `Generate a high-difficulty exam question that assesses a student's deep understanding of key concepts in ${state.input}, specifically targeting potential knowledge gaps identified in the student's historical interactions.
  Review their conversation history (${state.userData.history?.join('； ')}) to identify areas where they may have misconceptions or incomplete understanding.
  The question should require the student to critically analyze and synthesize their knowledge, applying it to a complex real-world problem or scenario.
  The task should challenge the student to demonstrate not only their theoretical understanding but also their ability to integrate and use the knowledge in practical, real-world contexts.
  After the question, provide a clear and concise answer that explains the key point and helps clarify the concept, with special attention to the student's academic background (${state.userData.grade ?? '博士生'}, ${state.userData.major_background ?? '具有丰富经验'}) and current grasp level (${state.userData.grasp_level ?? '十分熟练'}). Ouput these questions in ** Chinese **.`;
  return question_types.map((type) => {
    return new Send(type, {
      instructions: SYSTEM_PROMPT,
      descriptions: state.descriptions,
      graphData: state.graphData,
    });
  });
}

async function SingeChoice(state: typeof chainState.State) {
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
      content: state.instructions ?? '',
    },
    {
      role: 'human',
      content: `Using the knowledge extracted from the Neo4j query results, generate a Single-choice question with the following guidelines:
- Language: Chinese.
${state.descriptions !== '' ? `- Descriptions:  ${state.descriptions}` : ''}
- Question body: The question should focus on core concepts or key facts that require the user to recall and apply the information.
- Options: Provide four options labeled A, B, C, and D, with one correct answer and three distractors.
- Answer: Indicate the correct answer by specifying the corresponding option (A, B, C, or D).
- Difficulty level: Assign a difficulty level to the question based on the complexity of the content. 1 and 2 mean easy, 3 means medium, 4 and 5 mean hard.
- Explanation: Provide a clear and concise explanation of the correct answer, helping students understand why the answer is correct and why the other options are incorrect.
- Avoid extreme terms: Do not include extreme terms like “always” or “never” in the options, as they are easily ruled out by students.
- Question clarity: The question should be clear and concise, ensuring students can easily understand what is being asked.
- Knowledge alignment: Use the knowledge extracted from the Neo4j database as a reference point for creating questions, but feel free to incorporate related concepts or extend beyond the exact data provided.

Neo4j query results: ${state.graphData}
`,
    },
  ]);

  const enrichedResponse = {
    ...response,
    Type: 'SingleChoice',
    TypeText: '单选题',
    ProblemType: 1,
  };

  return { questions: enrichedResponse };
}

async function MultipleChoices(state: typeof chainState.State) {
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
      content: state.instructions ?? '',
    },
    {
      role: 'human',
      content: `Using the knowledge extracted from the Neo4j query results, generate a Multiple-choice question with the following guidelines:
- Language: Must be in Chinese.
${state.descriptions !== '' ? `- Descriptions:  ${state.descriptions}` : ''}
- Question body: The question should focus on core concepts or key facts that require the user to recall and apply the information.
- Options: Provide four options labeled A, B, C, and D, with at least two correct answer and other distractors.
- Answer: Indicate the correct answer by specifying the corresponding option (A, B, C, or D).
- Difficulty level: Assign a difficulty level to the question based on the complexity of the content. 1 and 2 mean easy, 3 means medium, 4 and 5 mean hard.
- Explanation: Provide a clear and concise explanation of these correct answer, helping students understand why the answer is correct and why the other options are incorrect.
- Avoid extreme terms: Do not include extreme terms like “always” or “never” in the options, as they are easily ruled out by students.
- Question clarity: The question should be clear and concise, ensuring students can easily understand what is being asked.
- Knowledge alignment: Use the knowledge extracted from the Neo4j database as a reference point for creating questions, but feel free to incorporate related concepts or extend beyond the exact data provided.

Neo4j query results: ${state.graphData}
`,
    },
  ]);

  const enrichedResponse = {
    ...response,
    Type: 'MultipleChoice',
    TypeText: '多选题',
    ProblemType: 2,
  };

  return { questions: enrichedResponse };
}

async function ShortAnswer(state: typeof chainState.State) {
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
      content: state.instructions ?? '',
    },
    {
      role: 'human',
      content: `Using the knowledge extracted from the Neo4j query results, generate a short-answer question with the following guidelines:
- Language: Must be in Chinese.
${state.descriptions !== '' ? `- Descriptions:  ${state.descriptions}` : ''}
- Question body: The question should focus on core concepts or key facts that require the user to recall and apply the information.
- Difficulty level: Assign a difficulty level to the question based on the complexity of the content. 1 and 2 mean easy, 3 means medium, 4 and 5 mean hard.
- Explanation: Provide a clear and concise explanation of this question that includes the key points that should be present in a complete answer. List the specific elements that would constitute a correct and complete response.
- Question clarity: The question should be clear and concise, ensuring students can easily understand what is being asked.
- Knowledge alignment: Use the knowledge extracted from the Neo4j database as a reference point for creating questions, but feel free to incorporate related concepts or extend beyond the exact data provided.

Neo4j query results: ${state.graphData}
`,
    },
  ]);

  const enrichedResponse = {
    ...response,
    Type: 'ShortAnswer',
    TypeText: '主观题',
    ProblemType: 5,
  };

  return { questions: enrichedResponse };
}

async function outputQuestions(state: typeof chainState.State) {
  return { output: state.questions };
}

const workflow = new StateGraph(chainState)
  .addNode('getGraph', getGraph)
  .addNode('SingeChoice', SingeChoice)
  .addNode('MultipleChoices', MultipleChoices)
  .addNode('ShortAnswer', ShortAnswer)
  .addNode('outputQuestions', outputQuestions)
  .addEdge('__start__', 'getGraph')
  .addConditionalEdges(
    'getGraph',
    async (state) => {
      const simpleQuestions = await routeSimpleQustions(state);
      const mediumQuestions = await routeMediumQustions(state);
      const hardQuestions = await routeHardQustions(state);
      return [...simpleQuestions, ...mediumQuestions, ...hardQuestions];
    },
    ['SingeChoice', 'MultipleChoices', 'ShortAnswer'],
  )
  .addEdge('SingeChoice', 'outputQuestions')
  .addEdge('MultipleChoices', 'outputQuestions')
  .addEdge('ShortAnswer', 'outputQuestions')
  .addEdge('outputQuestions', '__end__');

export const graph = workflow.compile({
  // if you want to update the state before calling the tools
  // interruptBefore: [],
});
