import { AIMessage, BaseMessage } from '@langchain/core/messages';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { RunnableToolLike } from '@langchain/core/runnables';
import { StructuredToolInterface } from '@langchain/core/tools';
import { Annotation, StateGraph } from '@langchain/langgraph';
import { ToolNode } from '@langchain/langgraph/prebuilt';
import { ChatOpenAI } from '@langchain/openai';
import SearchEsgTool from 'utils/tools/search_esg_tool';
import SearchInternalTool from 'utils/tools/search_internal_tool';
import SearchInternetTool from 'utils/tools/search_internet_tool';
import { z } from 'zod';
import './instrumentation.js';

const email = process.env.EMAIL ?? '';
const password = process.env.PASSWORD ?? '';
const openai_api_key = process.env.OPENAI_API_KEY ?? '';
const openai_chat_model = process.env.OPENAI_CHAT_MODEL ?? '';

const StateAnnotation = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (x, y) => x.concat(y),
  }),
  cycleCount: Annotation<number>(),
  contents: Annotation<string[]>({
    reducer: (_x, y) => [...y],
  }),
});

const tools: (StructuredToolInterface | RunnableToolLike)[] = [
  new SearchInternalTool({ email, password }),
  new SearchEsgTool({ email, password }),
  new SearchInternetTool({ email, password }),
];

const toolNode = new ToolNode(tools);

const maxCycleCount = 2;

// Define the function that calls the model
async function callModel(state: typeof StateAnnotation.State) {
  console.log('---      PROCESS: CALL MODEL      ---');

  const model = new ChatOpenAI({
    apiKey: openai_api_key,
    modelName: openai_chat_model,
    temperature: 0,
    streaming: false,
  }).bindTools(tools);

  const messages = state.messages;
  const lastMessage = [messages[messages.length - 1] as AIMessage];

  const prompt = ChatPromptTemplate.fromTemplate(
    `You are given a task from the following sources:\n
    -ali \n
    -others \n\n
    Based on the provided task description, select the most appropriate source to retrieve the information from the internal database.\n
    Meanwhile, using appropriate provided various tools to complete this task of composing a section for a report on the circular economy and its positive effects on carbon reduction.\n
    The retrieved results are allowed to be more than 5 and 10 at most to include more related information. \n
    Task description: {description}`,
  );
  // The retrieved results are allowed to be more than five. \n

  // const response = await model.invoke(lastMessage);
  const chain = prompt.pipe(model);
  const response = (await chain.invoke({ description: lastMessage })) as AIMessage;
  // console.log(response)
  let cycleCount = 1;
  if (state.cycleCount !== null) {
    cycleCount = state.cycleCount + 1;
  }

  // We return a list, because this will get added to the existing list
  return {
    messages: [response],
    cycleCount: cycleCount,
  };
}

function shouldContinue(state: typeof StateAnnotation.State) {
  const messages = state.messages;
  const lastMessage = messages[messages.length - 1] as AIMessage;

  // If the LLM makes a tool call, then we route to the "tools" node
  if (lastMessage.tool_calls?.length) {
    console.log('---     DECISION: CALL TOOLS      ---');
    return 'tools';
  }
  // Otherwise, we stop (reply to the user)
  console.log('---          DECISION: END        ---');
  return '__end__';
}

async function gradeContents(
  state: typeof StateAnnotation.State,
): Promise<Partial<typeof StateAnnotation.State>> {
  console.log('---    PROCESS: GET RELEVANCE     ---');

  const { messages } = state;

  const tool = {
    name: 'evaluate_contents_relevance',
    description:
      'Give a binary score of relevance and provide suggestions to improve the retrieved content.',
    schema: z.object({
      relevanceScore: z
        .string()
        .describe(
          "Binary score: 'yes' if the majority of the content is relevant to the query, 'no' otherwise.",
        ),
      suggestion: z
        .string()
        .optional()
        .describe('Suggestion for improving the retreival of more relevant contents'),
    }),
  };

  const prompt = ChatPromptTemplate.fromTemplate(
    ` Assign a binary score ('yes' or 'no') based on whether the majority of the content is relevant to the query.
      Yes: Most content is relevant.
      No: Content is irrelevant.
      Retrieved content:
        \n ------- \n
          {context} 
        \n ------- \n
      User query: {query} \n
      If the content is insufficient, suggest ways to improve retrieval.\n`,
  );

  const model = new ChatOpenAI({
    apiKey: openai_api_key,
    modelName: openai_chat_model,
    temperature: 0,
    streaming: false,
  }).bindTools([tool], {
    tool_choice: tool.name,
  });

  const chain = prompt.pipe(model);

  const retrievedContents: string[] = [];

  messages.forEach((item: BaseMessage) => {
    const data_j = JSON.stringify(item);
    const data = JSON.parse(data_j);
    if (data['type'] === 'constructor') {
      const id = data['id'] as string[];
      if (id[id.length - 1] === 'ToolMessage') {
        retrievedContents.push(data['kwargs']['content'] as string);
      }
    }
  });

  const score = (await chain.invoke({
    query: messages[0].content as string,
    context: retrievedContents.toString(),
  })) as AIMessage;

  return {
    messages: [score],
    contents: retrievedContents,
  };
}

// Define the function that determines whether to continue or not
// We can extract the state typing via `StateAnnotation.State`
function checkRelevance(state: typeof StateAnnotation.State) {
  console.log('---   PROCESS: CHECK RELEVANCE    ---');
  const { messages, cycleCount } = state;
  const lastMessage = messages[messages.length - 1] as AIMessage;
  // console.log(messages)
  if (lastMessage.tool_calls && lastMessage.tool_calls.length > 0) {
    if (lastMessage.tool_calls[0].args.relevanceScore === 'yes') {
      console.log('---  DECISION: CONTENTS RELEVANT  ---');
      return 'generate';
    } else {
      console.log('---DECISION: CONTENTS NOT RELEVANT---');
      if (cycleCount === maxCycleCount) {
        console.log('---  DECISION: MAX CYCLE REACHED  ---');
        return 'generate';
      } else {
        return 'supplement';
      }
    }
  } else {
    console.log('No tool calls found in the last message.');
    return '__end__';
  }
}

async function supplementContents(state: typeof StateAnnotation.State) {
  console.log('---  PROCESS: SUPPLEMENT CONTENT  ---');
  const messages = state.messages;
  const query = messages[0].content as string;
  const lastMessage = messages[messages.length - 1] as AIMessage;
  const suggestion = lastMessage.tool_calls
    ? (lastMessage.tool_calls[0].args.suggestion as string)
    : '';
  const prompt = ChatPromptTemplate.fromTemplate(
    `The previous content was insufficient for generating the report. \n
    Suggestions for improvement are provided below:\n 
    "{suggestion}"\n
    Based on these suggestions, design an improved query to retrieve more relevant information.\n
    Only return the refined query.\n`,
  );
  // Grader
  const model = new ChatOpenAI({
    apiKey: openai_api_key,
    modelName: openai_chat_model,
    temperature: 0,
    streaming: false,
  }).bindTools(tools);
  // const chain = prompt.pipe(model);
  const response = await prompt.pipe(model).invoke({ context: query, suggestion: suggestion });

  // console.log(response);

  return {
    contents: [response],
  };
}

async function generate(
  state: typeof StateAnnotation.State,
): Promise<Partial<typeof StateAnnotation.State>> {
  console.log('---    PROCESS: GENERATE ARTICLE  ---');
  const messages = state.messages;
  // console.log(messages);
  const query = messages[0].content as string;
  // const retrievedContents = messages.filter(msg => msg.name === 'ToolMessage').map(msg => msg.content as string);
  const retrievedContents: string[] = [];

  messages.forEach((item: BaseMessage) => {
    const data_j = JSON.stringify(item);
    const data = JSON.parse(data_j);
    if (data['type'] === 'constructor') {
      const id = data['id'] as string[];
      if (id[id.length - 1] === 'ToolMessage') {
        retrievedContents.push(data['kwargs']['content'] as string);
      }
    }
  });

  const prompt = ChatPromptTemplate.fromTemplate(
    `Write a clear, concise, and informative text to the user query using the retrieved content.\n
    Include specific examples containing statistical data, organizations, or locations from the content provided.\n
    User query: {query}\n
    Retrieved content: {retrievedContents}\n`,
  );

  const model = new ChatOpenAI({
    apiKey: openai_api_key,
    modelName: openai_chat_model,
    temperature: 0,
    streaming: false,
  });

  const ragChain = prompt.pipe(model);

  const response = await ragChain.invoke({
    query: query,
    retrievedContents: retrievedContents.toString(),
  });
  // console.log(response);

  return {
    messages: [response],
    contents: retrievedContents,
  };
}

async function suggestion(
  state: typeof StateAnnotation.State,
): Promise<Partial<typeof StateAnnotation.State>> {
  console.log('---  PROCESS: GIVING SUGGESTIONS  ---');

  const { messages } = state;

  const tool = {
    name: 'give_writing_suggestion',
    description: 'Judge whether a text is well-written and provide a suggestion.',
    schema: z.object({
      isWellWritten: z.string().describe("The text is well written? 'yes' or 'no'"),
      suggestion: z.string().optional().describe('Suggestion for how to improve the text'),
    }),
  };

  const prompt = ChatPromptTemplate.fromTemplate(
    `Evaluate the generated text based on the following criteria:\n
    1. Concise and informative.\n
    2. Clearly define the topic or themes, and provide relevant background and key information.\n
    3. Adequate supporting evidence from the latest sources, such as data, statistics, or examples.\n
    4. Contains detailed examples instead of general statements.\n
    5. Aligns with the language's conventions.\n
    6. Logically structured and easy to follow.\n
    7. Closely related to the retrieved content.\n
    8. Free from grammatical errors.\n
    Generated context : {context}\n
    User query: {query}\n
    Give a binary score ('yes' or 'no') to indicate if the text is well-written.\n
    If 'no', provide suggestions for improvement. \n`,
  );

  const model = new ChatOpenAI({
    apiKey: openai_api_key,
    modelName: openai_chat_model,
    temperature: 0,
    streaming: false,
  }).bindTools([tool], {
    tool_choice: tool.name,
  });

  const chain = prompt.pipe(model);

  const lastMessage = messages[messages.length - 1];

  const score = await chain.invoke({
    query: messages[0].content as string,
    context: lastMessage.content as string,
  });

  return {
    messages: [score],
  };
}

function checkWellWritten(state: typeof StateAnnotation.State) {
  console.log('---  PROCESS: CHECK WELL-WRITTEN  ---');
  const { messages, contents } = state;
  const lastMessage = messages[messages.length - 1] as AIMessage;
  const toolCalls = lastMessage.tool_calls;

  if (!toolCalls || !toolCalls.length) {
    throw new Error('Last message was not a function message');
  }

  if (toolCalls[0].args.isWellWritten === 'yes') {
    console.log('---  DECISION: TEXT WELL WRITTEN  ---');
    return '__end__';
  }
  console.log('---DECISION: TEXT NOT WELL WRITTEN---');
  return 'regenerate';
}

async function regenerate(
  state: typeof StateAnnotation.State,
): Promise<Partial<typeof StateAnnotation.State>> {
  console.log('---       PROCESS: REGENERATE     ---');

  const { messages } = state;
  const query = messages[0].content as string;
  const lastMessage = messages[messages.length - 1] as AIMessage;
  const toolCalls = lastMessage.tool_calls;

  if (!toolCalls || !toolCalls.length) {
    throw new Error('Last message was not a function message');
  }

  const suggestion = toolCalls[0].args.suggestion as string;

  const prompt = ChatPromptTemplate.fromTemplate(
    `The text generated before based on the user's query is poorly written. \n
      Please rewrite it using the following information:\n
      User query: \n{query}\n
      Retrieved contents: \n{retrievedContents}\n
      Improvement suggestions: \n{suggestion}\n
      Ensure that your rewrite directly addresses the improvement suggestions and enhances clarity, coherence, and overall quality of the text.\n`,
  );

  const model = new ChatOpenAI({
    apiKey: openai_api_key,
    modelName: openai_chat_model,
    temperature: 0,
    streaming: false,
  });

  const ragChain = prompt.pipe(model);

  const response = await ragChain.invoke({
    query: query,
    retrievedContents: state.contents.toString(),
    suggestion: suggestion,
  });

  return {
    messages: [response],
  };
}

// Define a new graph
const workflow = new StateGraph(StateAnnotation)
  .addNode('agent', callModel)
  .addNode('tools', toolNode)
  .addNode('grade', gradeContents)
  .addNode('supplement', supplementContents)
  .addNode('generate', generate)
  .addNode('suggestion', suggestion)
  .addNode('regenerate', regenerate)
  .addEdge('__start__', 'agent')
  .addConditionalEdges('agent', shouldContinue, ['__end__', 'tools'])
  .addEdge('tools', 'grade')
  .addConditionalEdges('grade', checkRelevance, ['generate', 'supplement'])
  .addEdge('supplement', 'agent')
  .addEdge('generate', 'suggestion')
  .addConditionalEdges('suggestion', checkWellWritten, ['__end__', 'regenerate'])
  .addEdge('regenerate', 'suggestion');

export const graph = workflow.compile({
  // if you want to update the state before calling the tools
  // interruptBefore: [],
});
