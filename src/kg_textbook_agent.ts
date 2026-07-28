import {
  LLMGraphTransformer,
  SYSTEM_PROMPT,
} from '@langchain/community/experimental/graph_transformers/llm';
import { Neo4jGraph } from '@langchain/community/graphs/neo4j_graph';
import { Document } from '@langchain/core/documents';
import { AIMessage } from '@langchain/core/messages';
import {
  ChatPromptTemplate,
  HumanMessagePromptTemplate,
  SystemMessagePromptTemplate,
} from '@langchain/core/prompts';
import { Annotation, Send, StateGraph } from '@langchain/langgraph';
import { ChatOpenAI } from '@langchain/openai';
import axios from 'axios';
import { z } from 'zod';
import './instrumentation.js';

// OPENAI
const openai_api_key = process.env.OPENAI_API_KEY ?? '';
const openai_chat_model = process.env.OPENAI_CHAT_MODEL ?? '';

// Neo4j database
const url = process.env.NEO4J_URI ?? '';
const username = process.env.NEO4J_USER ?? '';
const password = process.env.NEO4J_PASSWORD ?? '';

// Supabase
const base_url = process.env.BASE_URL ?? '';
const supabase_email = process.env.EMAIL ?? '';
const supabase_password = process.env.PASSWORD ?? '';
const supabase_authorization = process.env.SUPABASE_ANON_KEY ?? '';

type queryElement = {
  rec_id: string;
  textbook: string;
  chapter: string;
  query: string;
  content: string;
};

const chainState = Annotation.Root({
  inputs: Annotation<{ rec_id: string; title: string }[]>(),
  contents: Annotation<queryElement[]>({
    reducer: (x, y) => x.concat(y),
  }),
  knowledgeGraph: Annotation<queryElement[]>({
    reducer: (x, y) => x.concat(y),
  }),
});

async function routTextbooks(state: typeof chainState.State): Promise<Send[]> {
  console.log('---    PROCESS: ROUTE TEXTBOOKS    ---');
  return state.inputs.map((input) => {
    console.log('---- routeContent ----');
    return new Send('getChapters', {
      rec_id: input.rec_id,
      textbook: input.title,
      chapter: '',
      query: '',
      content: '',
    });
  });
}

async function getChapters(state: queryElement) {
  console.log('---    PROCESS: GET CHAPTERS    ---');
  // console.log(state)
  const { textbook, rec_id } = state;

  const query = '章节、目录、第 章、内容包含、章节设计';

  const url = `${base_url}/textbook_search`;
  const headers = {
    email: supabase_email,
    password: supabase_password,
    Authorization: `Bearer ${supabase_authorization}`,
    'Content-Type': 'application/json',
  };

  const payload = {
    query: query,
    filter: { rec_id: [rec_id] },
    topK: 5,
    extK: 2,
  };
  // console.log(payload)

  const chapter_response = await axios.post(url, payload, { headers });
  // console.log('---    PROCESS: GET CHAPTERS    ---');
  // console.log(chapter_response);

  const responseSchema = z.object({
    data: z.array(
      z.object({
        chapter: z
          .string()
          .describe('The list of chapters or contents extracted from the textbook.'),
        query: z.string().describe('The words or phrases related to the extracted content.'),
      }),
    ),
  });

  const tool = {
    name: 'Extract_Chapters',
    description: 'Extract the chapters or contents from the textbook.',
    schema: responseSchema,
  };

  const prompt = ChatPromptTemplate.fromTemplate(
    `Please carefully analyze the following text and extract its main content derived from its chapter structure. Summarize the key themes and essential information of each section as comprehensively as possible. When extracting, ensure that:
    - Each section or chapter is independent and complete, with no further subdivision possible. Excluding any 序言 or 绪论 or introduction or preface.
    - The core themes, key information, and main concepts are thoroughly summarized.
    - If there are chapter titles or structural markers, list them in order.
    - Exclude chapter numbers and any other irrelevant information from the extracted content.
    - The extracted content should reflect the overall logical framework of the text, not just surface-level information.
    - The extracted content should be in Chinese.
    - For each extracted content, provide 5 phrases or words with contextual relevance, considering "Who, What, When, Where, Why, and How", output as a string separated by commas.

    Here is the text to analyze: {chapter_response}`,
  );

  const model = new ChatOpenAI({
    apiKey: openai_api_key,
    modelName: openai_chat_model,
    temperature: 0,
    streaming: false,
  }).bindTools([tool]);

  const chain = prompt.pipe(model);

  const response = (await chain.invoke({
    chapter_response: chapter_response.data,
  })) as AIMessage;

  const data =
    response.tool_calls && response.tool_calls[0] && response.tool_calls[0].args
      ? (response.tool_calls[0].args.data as Array<{ chapter: string; query: string }>)
      : [];
  // console.log(data);
  const content: queryElement[] = [];
  data.map((element) => {
    // console.log(element);
    content.push({
      rec_id: rec_id,
      textbook: textbook,
      chapter: element.chapter,
      query: element.query,
      content: '',
    });
  });
  // console.log(content.length);
  return {
    contents: content,
  };
}

async function routeChapters(state: typeof chainState.State): Promise<Send[]> {
  console.log('---    PROCESS: ROUTE CHAPTERS    ---');
  // console.log(state.contents);
  return state.contents.map((element) => {
    console.log('---- routeContent ----');
    return new Send('getContents', {
      textbook: element.textbook,
      chapter: element.chapter,
      query: element.query,
      content: '',
    });
  });
}

async function getContents(state: queryElement): Promise<Partial<typeof chainState.State>> {
  console.log('---    PROCESS: GET CONTENTS    ---');

  const url = `${base_url}/textbook_search`;

  const headers = {
    email: supabase_email,
    password: supabase_password,
    Authorization: `Bearer ${supabase_authorization}`,
    'Content-Type': 'application/json',
  };

  const payload = {
    query: state.query,
    filter: { title: [state.textbook] },
    topK: 2,
  };

  const content_response = await axios.post(url, payload, { headers });

  const results = content_response.data as Array<{ content: string; source: string }>;

  let content = '';
  results.forEach((result) => {
    content += result.content;
  });

  const queryElement = {
    rec_id: state.rec_id,
    textbook: state.textbook,
    chapter: state.chapter,
    query: state.query,
    content: content,
  };

  return {
    knowledgeGraph: [queryElement],
  };
}

async function routeContents(state: typeof chainState.State): Promise<Send[]> {
  return state.knowledgeGraph.map((element: queryElement) => {
    console.log('---- routeContent ----');
    return new Send('generateKG', {
      textbook: element.textbook,
      chapter: element.chapter,
      content: element.content,
    });
  });
}

async function generateKG(state: queryElement) {
  console.log('---    PROCESS: GENERATE KG    ---');

  const graph = await Neo4jGraph.initialize({ url, username, password });

  const { textbook, chapter, content } = state;

  const system_message = SystemMessagePromptTemplate.fromTemplate(SYSTEM_PROMPT);

  const user_message = HumanMessagePromptTemplate.fromTemplate(
    `1. Given the ${textbook} ${chapter}, break down its knowledge structure from top to bottom into nodes and re-construct the complete tree-shape framework of a subject's content organized by internal logical relationships, showing hierarchy and interconnections. Note that, NODES are entities (DO NOT USE abstract terms like "knowledge modules"), and RELATIONSHIPS are connections between entities.
        The framework should be structured as follows:
          - The root node is the subject, which is the concise term of the ${textbook}.
          - The second level is topic that is the backbone of the subject.
          - The third level is main knowledge modules, which are key content areas under each theme.
          - The fourth level is knowledge units, which are coherent segments of learning content.
          - The fifth level is specific knowledge points, which are the smallest independent learning elements.
          - The sixth or more levels (if necessary) are sub-points to support the corresponding knowledge point.
          - NODES are entities (DO NOT USE abstract terms like "knowledge modules" and DO USE CONCISE WORD without abbreviations), and RELATIONSHIPS are connections between entities.
          - USING CHINESE LANGUAGE.
          - Converting this structure into triples to represent the relationships between nodes.
      2. Identify the node properties (more then one) for each subject, topic, knowledge module, knowledge units, knowledge points, and sub-points (if necessary). 
        2.1 KNOWLEDGE_TYPE
          - CONCEPT: definitions, terms, classifications, etc.
          - PRINCIPLE: laws, rules, theories, formulas, etc.
          - METHOD: procedures, techniques, approaches, etc.
          - SKILL: practical, abilities, operations, etc.
          - FACT: empirical data, observations, etc.
        2.2 CATEGORY: Identify a general field, INCLUDING BUT NOT LIMITED TO:
          - WATER
          - AIR
          - SOLID WASTE
          - SOLID AND LAND
          - ENERGY
          - BIODIVERSITY AND ECOSYSTEM
          - CHEMICAL AND TOXICOLOGICAL CONTAMINANTS
          - ENVIRONMENTAL MANAGEMENT AND POLICIES
          - INDUSTRIAL ECOLOGY
          - ENVIRONMENTAL MONITORING AND ANALYSIS
      3. Identify the relationships among topics, knowledge modules, knowledge units, knowledge points, and sub-points. 
        3.1 HIERARCHICAL RELATIONS: HAS_PART (whole-component relationship).
        3.2 PREREQUISITE:
          - REQUIRED: must learn before.
          - RECOMMENDED: better to learn before.
          - INDEPENDENT: can learn seperately.
        3.3 DEFINITIONAL RELATIONS (Dependencies between knowledge elements):
          - DEFINED_BY: formal definition.
          - EXEMPLIFIED_BY: concrete example.
          - CHARACTERIZED_BY: key features.
          - MEASURED_BY: quantification method.
          - REPRESENTED_BY: symbolic representation.
          `,
  );

  const prompt = ChatPromptTemplate.fromMessages([system_message, user_message]);

  const model = new ChatOpenAI({
    apiKey: openai_api_key,
    modelName: openai_chat_model,
    temperature: 0,
    streaming: false,
  });

  const llmGraphTransformerFiltered = new LLMGraphTransformer({
    llm: model,
    prompt: prompt,
    allowedNodes: ['CONCEPT', 'METHOD', 'PRINCIPLE', 'SKILL'],
    allowedRelationships: [
      'HAS_PART',
      'REQUIRED',
      'RECOMMENDED',
      'INDEPENDENT',
      'DEFINED_BY',
      'EXEMPLIFIED_BY',
      'CHARACTERIZED_BY',
      'MEASURED_BY',
      'REPRESENTED_BY',
    ],
    strictMode: false,
    nodeProperties: ['CATEGORY'],
  });

  const results = await llmGraphTransformerFiltered.convertToGraphDocuments([
    new Document({ pageContent: content }),
  ]);

  console.log('--    RESULTS    --');
  console.log(`Nodes: ${results[0].nodes.length}`);
  console.log(`Relationships:${results[0].relationships.length}`);

  await graph.addGraphDocuments(results);
}

const workflow = new StateGraph(chainState)
  .addNode('getChapters', getChapters)
  .addNode('generateKG', generateKG)
  .addNode('getContents', getContents)
  .addConditionalEdges('__start__', routTextbooks, ['getChapters'])
  .addConditionalEdges('getChapters', routeChapters, ['getContents'])
  .addConditionalEdges('getContents', routeContents, ['generateKG'])
  .addEdge('generateKG', '__end__');

export const graph = workflow.compile({
  // if you want to update the state before calling the tools
  // interruptBefore: [],
});
