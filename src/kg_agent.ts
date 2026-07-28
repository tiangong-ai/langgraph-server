import { ChatPromptTemplate } from '@langchain/core/prompts';
import { MessagesAnnotation, StateGraph } from '@langchain/langgraph';
import { ChatOpenAI } from '@langchain/openai';
import { z } from 'zod';
import './instrumentation.js';

const openai_api_key = process.env.OPENAI_API_KEY ?? '';
const openai_chat_model = process.env.OPENAI_CHAT_MODEL ?? '';

const nodeSchema = z
  .object({
    name: z
      .string()
      .describe(
        'The name of the core subject themes, knowledge modules, knowledge units, knowledge points.',
      ),
    node_id: z.string().describe('The UUID for the child node.'),
    children: z
      .array(z.lazy((): z.ZodTypeAny => nodeSchema))
      .describe(
        'List of child nodes of this node, between which there are hierarchical relationships.',
      ),
  })
  .describe('Schema for a child node.');

const relationSchema = z
  .object({
    relation_name: z.string().describe('The name of the relationship.'),
    source_node_id: z.string().describe('The UUID for the source node in the relationship.'),
    target_node_id: z.string().describe('The UUID for the target node in the relationship.'),
  })
  .describe('Logical relationships between nodes.');

const responseSchema = z
  .object({
    name: z.string().describe('The subject.'),
    node_id: z.string().describe('The UUID for this node.'),
    children: z
      .array(nodeSchema)
      .describe(
        'List of child terms of knowledge modules, knowledge units, knowledge points for the core subject themes.',
      ),
    relations: z
      .array(relationSchema)
      .describe(
        'List of relationships between the corresponding concepts associated with the node',
      ),
  })
  .describe('A tree-shape knowledge framework.');

async function generateKG(state: typeof MessagesAnnotation.State) {
  const messages = state.messages;
  const lastMessage = messages[messages.length - 1];
  const { context, topic, textbook } = JSON.parse(lastMessage.content as string);

  const tool = {
    name: 'generate_KG',
    description: 'Generate a knowledge graph from the given text.',
    schema: responseSchema,
  };

  const prompt = ChatPromptTemplate.fromTemplate(
    ` 
    Please analyze systematically the given text related to the topic of \' {topic} \' from {textbook} from top to bottom. \n

    Construct its knowledge system (in Chinese) to present the complete framework of a subject's content organized by internal logical relationships (showing hierarchy and interconnections).\n

    Please follow these guidelines:\n

    1. Using Python package to randomly generate a 20-character UUID (uuid version 4) for each knowledge points and relationship.\n
      - The UUID should follow this format: XXXXXXXXXXXXXXXXXXXX (20 characters with no hyphens and uppercase letters only).\n

    2. Identify the hierarchical knowledge structure (at least 4 levels), ensuring clear hierarchical relationships and logical connections between all levels. \n
      - Level 1: Subject, which is the {textbook} \n
      - Level 2: Topic, that forms the backbone of the subject. \n
      - Level 3: Main Knowledge Modules (key content areas under each theme)\n
      - Level 4: Knowledge Units (Coherent segments of learning content.)\n
      - Level 5: Specific Knowledge Points (The smallest independent learning element.)\n
      - Level 6: Sub-points (if necessary)\n

    3. Identify the following points for each topic:\n
      - List all knowledge modules under it.\n
      - Show logical relationships and learning sequence\n
      - Break down into knowledge units\n
      - Mark key/difficult points and dependencies\n
      - Decompose into specific knowledge points\n\n

    Text: {context}`,
  );

  const model = new ChatOpenAI({
    apiKey: openai_api_key,
    modelName: openai_chat_model,
    temperature: 0,
    streaming: false,
  }).bindTools([tool]);

  const chain = prompt.pipe(model);

  const response = (await chain.invoke({
    context: context,
    topic: topic,
    textbook: textbook,
  }));

  return {
    messages: response,
  };
}

const workflow = new StateGraph(MessagesAnnotation)
  .addNode('generateKG', generateKG)
  .addEdge('__start__', 'generateKG')
  .addEdge('generateKG', '__end__');

export const graph = workflow.compile({
  // if you want to update the state before calling the tools
  // interruptBefore: [],
});
