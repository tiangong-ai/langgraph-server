import { Annotation, StateGraph } from '@langchain/langgraph';
import { ChatOpenAI } from '@langchain/openai';
import axios from 'axios';
import neo4j from 'neo4j-driver';
import { z } from 'zod';
import './instrumentation.js';

// OpenAI
const openai_api_key = process.env.OPENAI_API_KEY ?? '';
const openai_chat_model_mini = process.env.OPENAI_CHAT_MODEL_MINI ?? '';

// Supabase
const base_url = process.env.BASE_URL ?? '';

type userElement = {
  grade?: string;
  major_background?: string;
  grasp_level?: string;
  major?: string;
  history?: string[];
};

type supabaseElement = {
  email: string;
  password: string;
  authorization: string;
};

const chainState = Annotation.Root({
  content: Annotation<string>(),
  knowledge_point: Annotation<string>(),
  learning_path: Annotation<{ name: string; id: string }[]>(),
  supabase_auth: Annotation<supabaseElement>(),
  userData: Annotation<userElement>(),
  refData: Annotation<{ content: string; source: string }>(),
  graphData: Annotation<string>(),
  pathData: Annotation<string[]>(),
});

async function getRefs(state: typeof chainState.State) {
  const url = `${base_url}/textbook_search`;
  const headers = {
    email: state.supabase_auth.email,
    password: state.supabase_auth.password,
    Authorization: `Bearer ${state.supabase_auth.authorization}`,
    'Content-Type': 'application/json',
  };

  const payload = {
    query: state.content + state.knowledge_point,
    topK: 1,
    extK: 2,
  };
  const textbook_chunks = await axios.post(url, payload, { headers });
  const textbook_chunks_data: { source: string; content: string }[] = textbook_chunks.data[0];

  return { refData: textbook_chunks_data };
}

async function getGraphData(state: typeof chainState.State) {
  const model = new ChatOpenAI({
    apiKey: openai_api_key,
    model: openai_chat_model_mini,
    streaming: false,
  });

  const pathResult = z.object({
    related_nodes: z.array(z.string().describe('Related Node Names')),
  });

  const structuredLlm = model.withStructuredOutput(pathResult);
  // and extract a knowledge structure centered around a single core concept. Reconstruct the content into a framework where all directly related knowledge points radiate outward from the core. Follow these detailed instructions:
  const response = await structuredLlm.invoke([
    {
      role: 'assistant',
      content: `Analyze the provided textbook chunks related to "${state.content}" as reference material to help construct a knowledge structure centered around "${state.knowledge_point}". While using these materials as guidance, focus on creating a comprehensive and logical knowledge framework. Follow these detailed instructions:
1. Framework Structure:
   - Root Node: The subject, represented by the concise term of "${state.knowledge_point}".
   - Related Nodes: Identify **only the most essential sub-concepts or components** that are direct constituents of ${state.knowledge_point}.
2. Node Selection Criteria:
   - Exclude tangentially related topics, broader categories, or solution concepts like "技术创新" or "管理方法"
   - Limit results to concrete characteristics rather than applications or management approaches
   - Keep the most critical and specific sub-properties
3. Node Representation:
   - Output each node as a slightly detailed Chinese phrase (not just a short keyword), using parenthesis style like "主题名（关键说明）".
   - Do not use colon forms such as "：" or ":" in node text.
   - Keep each node concise but informative; avoid overly long sentences.
   - Avoid abstract terms like "knowledge modules" in the node names
   - Output Format: Use **Chinese language** for key related node names.
   `,
    },
    {
      role: 'human',
      content: `Textbook chunks: ${state.refData.content}.`,
    },
  ]);

  const filteredNodes = response.related_nodes.filter((node) => node !== state.knowledge_point);

  return {
    graphData: filteredNodes,
  };
}

async function returnGraph(state: typeof chainState.State) {
  const url = process.env.NEO4J_URI ?? '';
  const username = process.env.NEO4J_USER ?? '';
  const password = process.env.NEO4J_PASSWORD ?? '';
  const driver = neo4j.driver(url, neo4j.auth.basic(username, password));
  const session = driver.session();

  try {
    const query_nodes = `
      UNWIND $list AS item
      MERGE (n:Concept {id: item})
      ON CREATE SET n.tag = "learning_path"
      RETURN n
    `;

    const result_nodes = await session.run(query_nodes, { list: state.graphData });
    const existingIds = new Set(state.learning_path.map((node) => node.id));
    const nodes: { name: string; id: string }[] = result_nodes.records
      .map((record) => {
        const node = record.get('n');
        return {
          name: node.properties.id,
          id: node.elementId,
        };
      })
      .filter((node) => !existingIds.has(node.id));

    const query_edges = `
      MATCH (a {id: $start_name})
      UNWIND $end_names AS end_name
      MATCH (b:Concept {id: end_name})
      WITH a, collect(b) AS related_nodes
      UNWIND related_nodes AS b
      MERGE (a)-[r:HAS_PART]->(b)
      ON CREATE SET r.tag = "learning_path"
    `;

    await session.run(query_edges, {
      start_name: state.knowledge_point,
      end_names: state.graphData,
    });

    return { pathData: nodes };
  } finally {
    await session.close();
    await driver.close();
  }
}

const workflow = new StateGraph(chainState)
  .addNode('getRefs', getRefs)
  .addNode('getGraphData', getGraphData)
  .addNode('returnGraph', returnGraph)
  .addEdge('__start__', 'getRefs')
  .addEdge('getRefs', 'getGraphData')
  .addEdge('getGraphData', 'returnGraph')
  .addEdge('returnGraph', '__end__');

export const graph = workflow.compile({
  // if you want to update the state before calling the tools
  // interruptBefore: [],
});
