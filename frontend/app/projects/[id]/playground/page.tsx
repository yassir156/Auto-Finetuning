"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import {
  getProject,
  getPlaygroundStatus,
  playgroundGenerate,
} from "@/lib/api";
import type {
  Project,
  PlaygroundStatusResponse,
  PlaygroundGenerateResponse,
} from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";

interface Message {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  tokens?: number;
  timestamp: Date;
}

export default function PlaygroundPage() {
  const params = useParams();
  const router = useRouter();
  const projectId = params.id as string;

  const [project, setProject] = useState<Project | null>(null);
  const [status, setStatus] = useState<PlaygroundStatusResponse | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [prompt, setPrompt] = useState("");
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Generation settings
  const [maxTokens, setMaxTokens] = useState(256);
  const [temperature, setTemperature] = useState(0.7);
  const [topP, setTopP] = useState(0.9);
  const [repetitionPenalty, setRepetitionPenalty] = useState(1.1);
  const [showSettings, setShowSettings] = useState(false);

  // Prompt template (instruction format)
  const [useTemplate, setUseTemplate] = useState(true);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Load project + playground status
  useEffect(() => {
    async function load() {
      try {
        const [proj, pgStatus] = await Promise.all([
          getProject(projectId),
          getPlaygroundStatus(projectId),
        ]);
        setProject(proj);
        setStatus(pgStatus);

        if (pgStatus.available) {
          setMessages([
            {
              id: "system-0",
              role: "system",
              content: `Model ready: ${proj.base_model_id} + LoRA adapter (${pgStatus.method?.toUpperCase() || "LoRA"})`,
              timestamp: new Date(),
            },
          ]);
        }
      } catch (err: any) {
        setError(err.detail || "Failed to load project");
      }
    }
    load();
  }, [projectId]);

  const handleGenerate = useCallback(async () => {
    if (!prompt.trim() || generating) return;

    const userPrompt = prompt.trim();
    setPrompt("");
    setError(null);

    // Format the prompt using the instruction template if enabled
    const formattedPrompt = useTemplate
      ? `### Instruction:\n${userPrompt}\n\n### Response:\n`
      : userPrompt;

    const userMessage: Message = {
      id: `user-${Date.now()}`,
      role: "user",
      content: userPrompt,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMessage]);
    setGenerating(true);

    try {
      const result: PlaygroundGenerateResponse = await playgroundGenerate(
        projectId,
        {
          prompt: formattedPrompt,
          max_new_tokens: maxTokens,
          temperature,
          top_p: topP,
          repetition_penalty: repetitionPenalty,
          do_sample: temperature > 0,
        },
      );

      const assistantMessage: Message = {
        id: `assistant-${Date.now()}`,
        role: "assistant",
        content: result.generated_text,
        tokens: result.num_tokens_generated,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err: any) {
      setError(err.detail || "Generation failed");
      const errorMessage: Message = {
        id: `error-${Date.now()}`,
        role: "system",
        content: `Error: ${err.detail || "Generation failed"}`,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setGenerating(false);
      textareaRef.current?.focus();
    }
  }, [prompt, generating, projectId, maxTokens, temperature, topP, repetitionPenalty, useTemplate]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleGenerate();
    }
  };

  const clearChat = () => {
    setMessages((prev) => prev.filter((m) => m.id === "system-0"));
  };

  if (error && !project) {
    return (
      <div className="max-w-4xl mx-auto space-y-6">
        <Card>
          <CardContent className="p-8 text-center">
            <p className="text-red-500">{error}</p>
            <Button className="mt-4" onClick={() => router.back()}>
              ← Back
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Playground</h1>
          <p className="text-sm text-muted-foreground">
            {project?.name || "Loading..."} — Test your fine-tuned model
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={() => router.push(`/projects/${projectId}`)}>
            ← Project
          </Button>
          {status?.available && (
            <Badge variant="default" className="bg-green-600">
              Model Ready
            </Badge>
          )}
          {status && !status.available && (
            <Badge variant="destructive">Not Available</Badge>
          )}
        </div>
      </div>

      {/* Model Info Bar */}
      {status?.available && (
        <Card>
          <CardContent className="py-3 px-4 flex flex-wrap items-center gap-4 text-sm">
            <div>
              <span className="text-muted-foreground">Base:</span>{" "}
              <span className="font-mono">{status.base_model_id}</span>
            </div>
            <div>
              <span className="text-muted-foreground">Method:</span>{" "}
              <Badge variant="outline">{status.method?.toUpperCase() || "LoRA"}</Badge>
            </div>
            {status.metrics?.train_loss != null && (
              <div>
                <span className="text-muted-foreground">Train Loss:</span>{" "}
                {(status.metrics.train_loss as number).toFixed(4)}
              </div>
            )}
            {status.metrics?.eval_loss != null && (
              <div>
                <span className="text-muted-foreground">Eval Loss:</span>{" "}
                {(status.metrics.eval_loss as number).toFixed(4)}
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Not Available State */}
      {status && !status.available && (
        <Card>
          <CardContent className="p-8 text-center space-y-4">
            <div className="text-4xl">🚧</div>
            <p className="text-lg font-medium">{status.message}</p>
            <p className="text-sm text-muted-foreground">
              Complete the training step first, then come back to test your model.
            </p>
            <Button onClick={() => router.push(`/projects/${projectId}`)}>
              Go to Project
            </Button>
          </CardContent>
        </Card>
      )}

      {/* Chat Area */}
      {status?.available && (
        <>
          {/* Settings Toggle */}
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowSettings(!showSettings)}
            >
              ⚙️ Settings {showSettings ? "▲" : "▼"}
            </Button>
            <label className="flex items-center gap-1.5 text-sm cursor-pointer">
              <input
                type="checkbox"
                checked={useTemplate}
                onChange={(e) => setUseTemplate(e.target.checked)}
                className="rounded"
              />
              Use Instruction Template
            </label>
            <Button variant="ghost" size="sm" onClick={clearChat}>
              🗑 Clear
            </Button>
          </div>

          {/* Settings Panel */}
          {showSettings && (
            <Card>
              <CardContent className="py-3 px-4 grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <label className="text-muted-foreground block mb-1">
                    Max Tokens
                  </label>
                  <input
                    type="number"
                    value={maxTokens}
                    onChange={(e) => setMaxTokens(Number(e.target.value))}
                    min={1}
                    max={2048}
                    className="w-full rounded border px-2 py-1 text-sm bg-background"
                  />
                </div>
                <div>
                  <label className="text-muted-foreground block mb-1">
                    Temperature
                  </label>
                  <input
                    type="number"
                    value={temperature}
                    onChange={(e) => setTemperature(Number(e.target.value))}
                    min={0}
                    max={2}
                    step={0.1}
                    className="w-full rounded border px-2 py-1 text-sm bg-background"
                  />
                </div>
                <div>
                  <label className="text-muted-foreground block mb-1">
                    Top-P
                  </label>
                  <input
                    type="number"
                    value={topP}
                    onChange={(e) => setTopP(Number(e.target.value))}
                    min={0}
                    max={1}
                    step={0.05}
                    className="w-full rounded border px-2 py-1 text-sm bg-background"
                  />
                </div>
                <div>
                  <label className="text-muted-foreground block mb-1">
                    Repetition Penalty
                  </label>
                  <input
                    type="number"
                    value={repetitionPenalty}
                    onChange={(e) => setRepetitionPenalty(Number(e.target.value))}
                    min={1}
                    max={3}
                    step={0.1}
                    className="w-full rounded border px-2 py-1 text-sm bg-background"
                  />
                </div>
              </CardContent>
            </Card>
          )}

          {/* Messages */}
          <Card className="min-h-[400px] max-h-[600px] overflow-y-auto">
            <CardContent className="p-4 space-y-4">
              {messages.map((msg) => (
                <div
                  key={msg.id}
                  className={`flex ${
                    msg.role === "user" ? "justify-end" : "justify-start"
                  }`}
                >
                  <div
                    className={`max-w-[85%] rounded-lg px-4 py-2 text-sm ${
                      msg.role === "user"
                        ? "bg-primary text-primary-foreground"
                        : msg.role === "system"
                        ? "bg-muted text-muted-foreground text-xs italic"
                        : "bg-muted"
                    }`}
                  >
                    <div className="whitespace-pre-wrap break-words">
                      {msg.content}
                    </div>
                    {msg.tokens != null && (
                      <div className="text-xs opacity-60 mt-1 text-right">
                        {msg.tokens} tokens
                      </div>
                    )}
                  </div>
                </div>
              ))}

              {generating && (
                <div className="flex justify-start">
                  <div className="bg-muted rounded-lg px-4 py-2 text-sm">
                    <div className="flex items-center gap-2">
                      <div className="animate-pulse">Generating...</div>
                      <div className="flex gap-1">
                        <div className="w-1.5 h-1.5 bg-foreground/40 rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                        <div className="w-1.5 h-1.5 bg-foreground/40 rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                        <div className="w-1.5 h-1.5 bg-foreground/40 rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
                      </div>
                    </div>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </CardContent>
          </Card>

          {/* Input */}
          <div className="flex gap-2">
            <Textarea
              ref={textareaRef}
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={
                useTemplate
                  ? "Enter your instruction... (Shift+Enter for new line)"
                  : "Enter raw prompt... (Shift+Enter for new line)"
              }
              className="min-h-[60px] max-h-[150px] resize-none"
              disabled={generating}
            />
            <Button
              onClick={handleGenerate}
              disabled={!prompt.trim() || generating}
              className="self-end px-6"
            >
              {generating ? "..." : "Send"}
            </Button>
          </div>

          {/* Usage hint */}
          {useTemplate && (
            <p className="text-xs text-muted-foreground">
              💡 Your text will be wrapped in <code>### Instruction: ... ### Response:</code> template.
              Uncheck &quot;Use Instruction Template&quot; to send raw prompts.
            </p>
          )}

          {error && (
            <p className="text-sm text-red-500">{error}</p>
          )}
        </>
      )}
    </div>
  );
}
