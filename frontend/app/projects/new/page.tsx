"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { createProject, getTaskTypes } from "@/lib/api";
import type { TaskTypeConfig } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function NewProjectPage() {
  const router = useRouter();
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [taskType, setTaskType] = useState("instruction_tuning");
  const [taskTypes, setTaskTypes] = useState<TaskTypeConfig[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getTaskTypes().then((res) => setTaskTypes(res.task_types)).catch(() => {});
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) {
      setError("Project name is required");
      return;
    }

    try {
      setLoading(true);
      setError(null);
      const project = await createProject({
        name: name.trim(),
        description: description.trim() || undefined,
        task_type: taskType,
      });
      router.push(`/projects/${project.id}`);
    } catch (err: unknown) {
      const msg =
        err && typeof err === "object" && "detail" in err
          ? (err as { detail: string }).detail
          : "Failed to create project";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mx-auto max-w-lg">
      <Card>
        <CardHeader>
          <CardTitle>Create New Project</CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="mb-1 block text-sm font-medium">
                Project Name *
              </label>
              <Input
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="My Fine-Tuning Project"
                maxLength={200}
              />
            </div>

            <div>
              <label className="mb-1 block text-sm font-medium">
                Description
              </label>
              <Textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Optional project description..."
                maxLength={2000}
                rows={3}
              />
            </div>

            <div>
              <label className="mb-1 block text-sm font-medium">
                Task Type *
              </label>
              <div className="grid gap-2 sm:grid-cols-2">
                {taskTypes.map((tt) => {
                  const isSelected = taskType === tt.key;
                  return (
                    <button
                      key={tt.key}
                      type="button"
                      onClick={() => setTaskType(tt.key)}
                      className={`rounded-lg border p-3 text-left transition-colors ${
                        isSelected
                          ? "border-primary bg-primary/5 ring-2 ring-primary"
                          : "border-border hover:border-primary/50 hover:bg-muted/50"
                      }`}
                    >
                      <div className="font-medium text-sm">{tt.label}</div>
                      <p className="mt-0.5 text-xs text-muted-foreground">
                        {tt.description}
                      </p>
                    </button>
                  );
                })}
              </div>
            </div>

            {error && (
              <p className="text-sm text-destructive">{error}</p>
            )}

            <div className="flex gap-3">
              <Button type="submit" disabled={loading} className="flex-1">
                {loading ? "Creating..." : "Create Project"}
              </Button>
              <Button
                type="button"
                variant="outline"
                onClick={() => router.push("/")}
              >
                Cancel
              </Button>
            </div>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
