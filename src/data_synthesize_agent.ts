import { Annotation, Send, StateGraph } from '@langchain/langgraph';
import { ChatOpenAI } from '@langchain/openai';
import axios from 'axios';
import { z } from 'zod';
import './instrumentation.js';

// OpenAI
const openai_chat_model = process.env.OPENAI_CHAT_MODEL ?? '';
const openai_api_key = process.env.OPENAI_API_KEY ?? '';

// Supabase
const base_url = process.env.BASE_URL ?? '';

const chainState = Annotation.Root({
  input: Annotation<string>(),
  textbook_chunks: Annotation<{ content: string; source: string }[]>(),
  chunk_content: Annotation<string>(),
  chunk_analysis: Annotation<string[]>({
    reducer: (x, y) => x.concat(y),
  }),
  supabase_email: Annotation<string>(),
  supabase_password: Annotation<string>(),
  supabase_authorization: Annotation<string>(),
});

async function searchTextbooks(state: typeof chainState.State) {
  const url = `${base_url}/textbook_search`;
  const headers = {
    email: state.supabase_email,
    password: state.supabase_password,
    Authorization: `Bearer ${state.supabase_authorization}`,
    'Content-Type': 'application/json',
  };

  const payload = {
    query: state.input,
    topK: 3,
    extK: 2,
  };
  const textbook_chunks = await axios.post(url, payload, { headers });

  return { textbook_chunks: textbook_chunks.data };
}

async function routeChunks(state: typeof chainState.State): Promise<Send[]> {
  return state.textbook_chunks.map((chunk: { content: string; source: string }) => {
    return new Send('analyzeChunk', {
      chunk_content: chunk.content,
    });
  });
}

async function analyzeChunk(state: typeof chainState.State) {
  const responseSchema = z.object({
    analysis: z.array(
      z.object({
        question: z.string().describe('问题内容与支撑材料'),
        response: z.string().describe('详细解答与明确结论'),
        chain_of_thought: z.string().describe('思考过程以及用于分析的材料'),
      }),
    ),
  });

  const model = new ChatOpenAI({
    apiKey: openai_api_key,
    modelName: openai_chat_model,
    streaming: false,
    temperature: 0,
  });

  const structuredLlm = model.withStructuredOutput(responseSchema);

  const response = await structuredLlm.invoke([
    {
      role: 'system',
      content: `请以一位资深教授的角度，基于所提供的教材内容，完成以下三个任务。所有解释、分析和思考过程均以中文输出。
1.明确以下教材内容中所有的教学目标、教学重点和教学难点（尽可能覆盖全面），并以完整的学术问题形式提出。每个问题的表述需涵盖核心概念，并强调“反应机理、影响因素及其在环境工程中的应用”三方面内容（如适用）。同时，为每个问题提供支撑材料（如案例、数据、理论依据、相关文献等）。
2.针对第一步中提出的每个问题，结合教材内容进行详细解答并给出明确结论。回答内容需尽可能详细、全面且准确。
-思考过程需层层递进，明确问题的背景、核心原理、分析方法、数据支持、计算方式（若适用）以及现实应用。
-使用系统性的推理方式（如因果分析、数理推导、对比分析等）来组织答案，而不仅是总结教材内容。
3.对于每个教学目标、教学重点和教学难点，根据提供内容完整阐述你的思考过程（chain-of-thought），重点强调如何科学、系统地思考和分析问题。具体要求如下：
-明确问题的认知路径：从问题的背景入手，分析其学术价值或实际意义，并拆解为子问题。
-建立逻辑推理链：围绕问题，依次思考核心概念、影响因素、适用条件及其内在联系。
-提供完整支撑材料，而非简单总结教材内容（数据以表格展示，计算方式以完整公式表述，案例以详细描述呈现，而非仅提供来源）。`,
    },
    {
      role: 'human',
      content: state.chunk_content ?? '',
    },
  ]);

  return { chunk_analysis: response.analysis };
}

const workflow = new StateGraph(chainState)
  .addNode('searchTextbooks', searchTextbooks)
  .addNode('analyzeChunk', analyzeChunk)
  .addEdge('__start__', 'searchTextbooks')
  .addConditionalEdges('searchTextbooks', routeChunks, ['analyzeChunk'])
  .addEdge('analyzeChunk', '__end__');

export const graph = workflow.compile({
  // if you want to update the state before calling the tools
  // interruptBefore: [],
});
