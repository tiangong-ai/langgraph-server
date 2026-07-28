import { suggestData } from '@tiangong-lca/tidas-sdk';
import { StateGraph,Annotation } from '@langchain/langgraph';
import './instrumentation.js';
// import { z } from 'zod';
// import { BaseMessage } from '@langchain/core/messages';


// const inputSchema = z.object({
//   data: z.any(),
//   dataType: z.string(),
//   options:z.any(),
// });

// const outputSchema = z.object({
//   data: z.any(),
//   diffSummary: z.string().optional(),
//   diffHTML: z.string().optional(),
//   success: z.boolean(),
//   error: z.string().optional(),
// });


export const InternalStateAnnotation = Annotation.Root({
  // input: Annotation<z.infer<typeof inputSchema>>(),
  data: Annotation<any>(),
  dataType: Annotation<string>(),
  options: Annotation<any>(),
});

export const OutputStateAnnotation = Annotation.Root({
  // output: Annotation<z.infer<typeof outputSchema>>(),
  data: Annotation<any>(),
  dataType: Annotation<string>(),
  options: Annotation<any>(),
  diffSummary: Annotation<string>(),
  diffHTML: Annotation<string>(),
  success: Annotation<boolean>(),
  error: Annotation<string>(),
});

const openai_api_key = process.env.OPENAI_API_KEY ?? '';
const openai_chat_model = process.env.OPENAI_CHAT_MODEL ?? '';
const openai_base_url = process.env.OPENAI_BASE_URL ?? undefined;

const defaultOptions = {
  maxRetries: 3,
  outputDiffSummary: true,
  outputDiffHTML: true,
  modelConfig: {
    model: openai_chat_model,
    apiKey: openai_api_key,
    baseURL: openai_base_url,
  },
};
// console.log('defaultOptions');
// console.log(JSON.stringify(defaultOptions));

const suggestDataNode = async (state: typeof InternalStateAnnotation.State): Promise<Partial<typeof OutputStateAnnotation.State>>  => {
  const timeStart = Date.now();
  const response = await suggestData(state.data, state.dataType, {
    ...defaultOptions,
    ...state.options,
  });
  const timeEnd = Date.now();
  console.log(`SuggestDataNode: Time taken: ${timeEnd - timeStart} milliseconds`);
  // console.log('response');
  // console.log(JSON.stringify(response));
  return {  data: response.data, dataType: state.dataType, options: state.options, diffSummary: response.diffSummary, diffHTML: response.diffHTML, success: response.success, error: response.error};
};


const workflow = new StateGraph({input: InternalStateAnnotation, output: OutputStateAnnotation})
  .addNode('agent', suggestDataNode)
  .addEdge('__start__', 'agent')
  .addEdge('agent', '__end__');

export const graph = workflow.compile({});
