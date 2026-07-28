import { BaseMessage } from '@langchain/core/messages';
import { ChatOpenAI } from '@langchain/openai';
import './instrumentation.js';

import { Annotation, Command, StateGraph } from '@langchain/langgraph';

import { Calculator } from 'utils/nodes/agent_calculating';
import { Reasoning } from 'utils/nodes/agent_reasoning';

import { z } from 'zod';

const openai_api_key = process.env.OPENAI_API_KEY ?? '';
const openai_chat_model = process.env.OPENAI_CHAT_MODEL ?? '';

const StateAnnotation = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (x, y) => x.concat(y),
  }),
  answers: Annotation<string[]>({
    reducer: (x, y) => (y ? x.concat(y) : x),
  }),
  suggestion: Annotation<string>(),
  score: Annotation<number>(),
});

const allocateTask = async (state: typeof StateAnnotation.State) => {
  console.log('----  allocateTask  ----');

  const taskType = z.object({
    taskType: z
      .enum(['Calculation', 'Reasoning', 'Wikipedia'])
      .describe('The type of task to be performed by the model in the next step.'),
  });

  const model = new ChatOpenAI({
    apiKey: openai_api_key,
    modelName: openai_chat_model,
    streaming: false,
  });
  // this is a replacement for a real conditional edge function
  const structuredLlm = model.withStructuredOutput(taskType);

  const response = await structuredLlm.invoke([
    {
      role: 'human',
      content: `You are an intelligent assistant skilled in analyzing and categorizing various types of problems. Given a problem statement, classify it into one of the following categories:
- Calculation: The problem requires performing specific mathematical computations or executing code. If the problem only involves explaining the logical steps or methodology for a calculation (without actual computation), classify it as Reasoning.
- Reasoning: The problem focuses on understanding concepts, logical thinking, or solving issues based on specialized knowledge (e.g., standards, academic papers, or reports). This includes explaining methodologies or theoretical concepts without performing calculations.
- Wikipedia: The problem seeks simple, factual information that can be directly looked up or summarized from general sources like Wikipedia.`,
    },
    state.messages[0],
  ]);
  // note how Command allows you to BOTH update the graph state AND route to the next node
  return new Command({
    // this is a replacement for an edge
    goto: response.taskType,
  });
};

async function runCalculation(state: typeof StateAnnotation.State) {
  console.log('---- runCalculation ----');
  const { messages, answers } = await Calculator.invoke({
    messages: [state.messages[0]],
    suggestion: state.suggestion ? state.suggestion : '',
  });
  return {
    messages: messages,
    answers: answers,
  };
}

async function runReasoning(state: typeof StateAnnotation.State) {
  console.log('---- runReasoning ----');
  const { messages, answers } = await Reasoning.invoke({
    messages: [state.messages[0]],
    suggestion: state.suggestion ? state.suggestion : '',
  });
  return {
    messages: messages,
    answers: answers,
  };
}

async function runAnswering(state: typeof StateAnnotation.State) {
  console.log('---- runAnswering ----');
  const model_api_key = process.env.BAIDU_API_KEY ?? '';
  const chat_model = process.env.BAIDU_CHAT_MODEL ?? '';
  const base_url = process.env.BAIDU_BASE_URL ?? '';

  const model = new ChatOpenAI({
    apiKey: model_api_key,
    modelName: chat_model,
    streaming: false,
    configuration: {
      baseURL: base_url,
    },
  });

  const response = await model.invoke([
    {
      role: 'human',
      content: `You are an environmental science expert tasked with answering questions. Please follow these guidelines to answer the problem.
1. Read the Problem Carefully: Understand the question and determine whether it requires logical reasoning or factual knowledge.
2. Answer Appropriately:
- For Reasoning: Provide a step-by-step logical explanation, deducing the answer from the given information.
- For Knowledge: Provide a direct, fact-based answer or explanation relevant to the subject matter.
3. Be Clear and Concise: Ensure the answer is well-structured and precise.
4. Language: use the same language as the question.
${state.suggestion !== '' ? `*** Suggestions ***  ${state.suggestion}` : ''}`,
    },
    ...state.messages,
  ]);
  return {
    messages: response,
    answers: [response.lc_kwargs.content],
  };
}

async function evaluateAnswer(state: typeof StateAnnotation.State) {
  console.log('--- evaluateOutput ---');
  // console.log(state.messages);
  const firstMessage = state.messages[0];
  const answer: string = state.answers[state.answers.length - 1];
  console.log('---- answer ----');
  console.log(answer);

  const scoreResult = z.object({
    score: z
      .number()
      .describe(
        'This score ranging from 0 to 30 indicates how well the response meets the evaluate criteria, with 30 being the perfect answer while 0 being the worst.',
      ),
    suggestion: z
      .string()
      .optional()
      .describe('Suggestion for improving the approachs to problem solving'),
  });

  const evaluationCriteria = `
        *** Logical Coherence (totally 20 points) *** 
        - Structure: Is the analysis logically structured, with a clear and complete flow?
          10–9 points: Highly logical and well-structured
          8–7 points: Mostly clear, with minor flaws
          6–5 points: Noticeable gaps or missing steps
          4–3 points: Poor structure or incomplete reasoning
          2–1 points: Chaotic or entirely illogical
        - Use of Conditions: Are all provided conditions effectively used?
          10–9 points: Fully and correctly utilized
          8–7 points: Mostly utilized, with minor omissions
          6–5 points: Some conditions not effectively used
          2–1 points: Misused or ignored

        *** Clarity of Results (10 points) *** 
        - If this is a computation qustion: Are the numerical results accurate, clear, and logically consistent?
          10–9 points: Clear, accurate, and rigorous
          8–7 points: Mostly correct, with minor inaccuracies
          6–5 points: Significant errors or unclear presentation
          2–1 points: Incorrect or irrelevant results
        - Else, for other questions (responses require qualitative Insights): Are the explanations and reasoning clear and insightful?
          10–9 points: Clear, insightful, and well-supported
          8–7 points: Generally sound but somewhat superficial
          6–5 points: Lacks depth or has inconsistencies
          2–1 points: Weak or missing analysis
  `;

  const model = new ChatOpenAI({
    apiKey: openai_api_key,
    modelName: openai_chat_model,
    streaming: false,
  });

  const structuredLlm = model.withStructuredOutput(scoreResult);

  const response = await structuredLlm.invoke([
    {
      role: 'human',
      content: `Please evaluate the responses to the following question, focusing on logical coherence and whether it provides clear and reasonable results, and provide suggestion concerning evaluation criteria to improve the quality of the response. The evaluation criteria are:\n\n
      Criteria: ${evaluationCriteria}.
      Question: ${firstMessage.content}.
      Response: ${answer}.`,
    },
  ]);

  return {
    answers: state.answers,
    messages: response,
    suggestion: response?.suggestion ?? '',
    score: response.score,
  };
}

function routeModelThink(state: typeof StateAnnotation.State) {
  console.log('--- routeModelThink ---');
  if (state.score > 25) {
    console.log('---- finalResult ----');
    console.log(state.answers[state.answers.length - 1]);
    return '__end__';
  }
  console.log('---- getAnswer ----');
  return 'allocateTask';
}

const workflow = new StateGraph(StateAnnotation)
  .addNode('allocateTask', allocateTask, { ends: ['Calculation', 'Reasoning', 'Wikipedia'] })
  .addNode('Calculation', runCalculation)
  .addNode('Reasoning', runReasoning)
  .addNode('Wikipedia', runAnswering)
  .addNode('evaluateAnswer', evaluateAnswer)
  .addEdge('__start__', 'allocateTask')
  .addEdge('Calculation', 'evaluateAnswer')
  .addEdge('Reasoning', 'evaluateAnswer')
  .addEdge('Wikipedia', 'evaluateAnswer')
  .addConditionalEdges('evaluateAnswer', routeModelThink, ['allocateTask', '__end__']);

export const graph = workflow.compile({
  // if you want to update the state before calling the tools
  // interruptBefore: [],
});
