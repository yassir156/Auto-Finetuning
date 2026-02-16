"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { listProjects, deleteProject } from "@/lib/api";
import type { Project } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { StatusBadge } from "@/components/status-badge";

export default function Home() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchProjects = async () => {
    try {
      setLoading(true);
      const res = await listProjects({ limit: 50, offset: 0 });
      setProjects(res.items);
    } catch {
      /* ignore */
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchProjects();
  }, []);

  const handleDelete = async (id: string) => {
    if (!confirm("Delete this project?")) return;
    try {
      await deleteProject(id);
      setProjects((prev) => prev.filter((p) => p.id !== id));
    } catch {
      /* ignore */
    }
  };

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold tracking-tight">Projects</h1>
        <Link href="/projects/new">
          <Button>+ New Project</Button>
        </Link>
      </div>

      {loading ? (
        <div className="rounded-lg border bg-card p-8 text-center text-muted-foreground">
          Loading projects...
        </div>
      ) : projects.length === 0 ? (
        <div className="rounded-lg border bg-card p-8 text-center text-muted-foreground">
          <p>No projects yet. Create one to get started.</p>
        </div>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {projects.map((project) => (
            <Card key={project.id} className="flex flex-col">
              <CardHeader className="pb-3">
                <div className="flex items-start justify-between">
                  <CardTitle className="text-lg">
                    <Link
                      href={`/projects/${project.id}`}
                      className="hover:underline"
                    >
                      {project.name}
                    </Link>
                  </CardTitle>
                  <StatusBadge status={project.status} />
                </div>
                {project.description && (
                  <p className="text-sm text-muted-foreground line-clamp-2">
                    {project.description}
                  </p>
                )}
              </CardHeader>
              <CardContent className="flex-1">
                <div className="flex flex-wrap gap-2 text-xs text-muted-foreground">
                  <span>Task: {project.task_type}</span>
                  {project.base_model_id && (
                    <span>Model: {project.base_model_id}</span>
                  )}
                </div>
                <p className="mt-2 text-xs text-muted-foreground">
                  Created: {new Date(project.created_at).toLocaleDateString()}
                </p>
              </CardContent>
              <div className="flex items-center gap-2 border-t p-4">
                <Link href={`/projects/${project.id}`} className="flex-1">
                  <Button variant="outline" size="sm" className="w-full">
                    Open
                  </Button>
                </Link>
                <Button
                  variant="ghost"
                  size="sm"
                  className="text-destructive hover:text-destructive"
                  onClick={() => handleDelete(project.id)}
                >
                  Delete
                </Button>
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
