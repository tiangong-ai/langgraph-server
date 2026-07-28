import { LangChainInstrumentation } from '@arizeai/openinference-instrumentation-langchain';
import * as CallbackManagerModule from '@langchain/core/callbacks/manager';
import { resourceFromAttributes } from '@opentelemetry/resources';
import {
  BatchSpanProcessor,
  type ReadableSpan,
  type SpanExporter,
} from '@opentelemetry/sdk-trace-base';
import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';

class DeferredFilesSpanExporter implements SpanExporter {
  private exporterPromise: Promise<SpanExporter> | undefined;

  export(spans: ReadableSpan[], resultCallback: Parameters<SpanExporter['export']>[1]): void {
    void this.getExporter().then(
      (exporter) => exporter.export(spans, resultCallback),
      (error: unknown) =>
        resultCallback({
          code: 1,
          error: error instanceof Error ? error : new Error(String(error)),
        }),
    );
  }

  async shutdown(): Promise<void> {
    if (this.exporterPromise) {
      await (await this.exporterPromise).shutdown();
    }
  }

  private getExporter(): Promise<SpanExporter> {
    this.exporterPromise ??= import('@agentpond/files-sdk/otel').then(
      ({ createFilesSpanExporterFromRuntimeEnv }) => createFilesSpanExporterFromRuntimeEnv(),
    );
    return this.exporterPromise;
  }
}

let tracerProvider: NodeTracerProvider | undefined;
let shutdownPromise: Promise<void> | undefined;

function reportShutdownError(error: unknown): void {
  console.error(
    '[AgentPond] Failed to flush tracing during shutdown:',
    error instanceof Error ? error.message : String(error),
  );
}

function registerTerminationHandler(signal: NodeJS.Signals): void {
  process.prependOnceListener(signal, () => {
    const hostOwnsShutdown = process.listenerCount(signal) > 0;

    void shutdownAgentPondTracing()
      .catch(reportShutdownError)
      .finally(() => {
        if (!hostOwnsShutdown) {
          process.kill(process.pid, signal);
        }
      });
  });
}

if (process.env.FILES_SDK_PROVIDER) {
  tracerProvider = new NodeTracerProvider({
    resource: resourceFromAttributes({
      'service.name': 'tiangong-ai-langgraph-server',
    }),
    spanProcessors: [new BatchSpanProcessor(new DeferredFilesSpanExporter())],
  });
  tracerProvider.register();

  const instrumentation = new LangChainInstrumentation({ tracerProvider });
  instrumentation.manuallyInstrument(CallbackManagerModule);

  process.once('beforeExit', () => {
    void shutdownAgentPondTracing().catch(reportShutdownError);
  });
  registerTerminationHandler('SIGINT');
  registerTerminationHandler('SIGTERM');
}

export function shutdownAgentPondTracing(): Promise<void> {
  if (!tracerProvider) return Promise.resolve();
  shutdownPromise ??= tracerProvider.shutdown();
  return shutdownPromise;
}
