/**
 * Logger utility para el frontend
 * Proporciona logging estructurado con diferentes niveles
 */

export enum LogLevel {
  DEBUG = 0,
  INFO = 1,
  WARN = 2,
  ERROR = 3,
}

class Logger {
  private level: LogLevel;
  private context: string;

  constructor(context: string = "App", level: LogLevel = LogLevel.INFO) {
    this.context = context;
    this.level = level;
  }

  private shouldLog(level: LogLevel): boolean {
    return level >= this.level;
  }

  private formatMessage(level: string, message: string, ...args: unknown[]): void {
    const timestamp = new Date().toISOString();
    const formattedArgs = args.length > 0 ? JSON.stringify(args) : "";
    console.log(
      `[${timestamp}] [${level}] [${this.context}] ${message}${formattedArgs ? ` ${formattedArgs}` : ""}`
    );
  }

  debug(message: string, ...args: unknown[]): void {
    if (this.shouldLog(LogLevel.DEBUG)) {
      this.formatMessage("DEBUG", message, ...args);
    }
  }

  info(message: string, ...args: unknown[]): void {
    if (this.shouldLog(LogLevel.INFO)) {
      this.formatMessage("INFO", message, ...args);
    }
  }

  warn(message: string, ...args: unknown[]): void {
    if (this.shouldLog(LogLevel.WARN)) {
      this.formatMessage("WARN", message, ...args);
    }
  }

  error(message: string, error?: Error | unknown, ...args: unknown[]): void {
    if (this.shouldLog(LogLevel.ERROR)) {
      this.formatMessage("ERROR", message, ...args);
      if (error instanceof Error) {
        console.error("Error details:", {
          message: error.message,
          stack: error.stack,
          name: error.name,
        });
      } else if (error) {
        console.error("Error details:", error);
      }
    }
  }
}

// Logger por defecto
export const logger = new Logger(
  "App",
  import.meta.env.DEV ? LogLevel.DEBUG : LogLevel.INFO
);

// Factory para crear loggers con contexto
export function createLogger(context: string): Logger {
  return new Logger(context, import.meta.env.DEV ? LogLevel.DEBUG : LogLevel.INFO);
}
