"use client";

import { useCallback, useRef, useState } from "react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";

interface FileUploadDropzoneProps {
  onFiles: (files: File[]) => void;
  accept?: string;
  maxFiles?: number;
  disabled?: boolean;
  className?: string;
}

export function FileUploadDropzone({
  onFiles,
  accept = ".pdf,.docx,.txt,.md,.jsonl,.csv,.json",
  maxFiles = 20,
  disabled = false,
  className,
}: FileUploadDropzoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      if (disabled) return;
      const files = Array.from(e.dataTransfer.files).slice(0, maxFiles);
      if (files.length > 0) onFiles(files);
    },
    [onFiles, maxFiles, disabled]
  );

  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = Array.from(e.target.files ?? []).slice(0, maxFiles);
      if (files.length > 0) onFiles(files);
      e.target.value = "";
    },
    [onFiles, maxFiles]
  );

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      className={cn(
        "relative flex flex-col items-center justify-center rounded-lg border-2 border-dashed p-8 transition-colors",
        isDragging
          ? "border-primary bg-primary/5"
          : "border-muted-foreground/25 hover:border-muted-foreground/50",
        disabled && "cursor-not-allowed opacity-50",
        className
      )}
    >
      <svg
        className="mb-3 h-10 w-10 text-muted-foreground"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={1.5}
          d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
        />
      </svg>
      <p className="mb-1 text-sm font-medium">
        Drag &amp; drop files here
      </p>
      <p className="mb-3 text-xs text-muted-foreground">
        PDF, DOCX, TXT, MD, JSONL, CSV, JSON (max {maxFiles} files)
      </p>
      <Button
        variant="outline"
        size="sm"
        disabled={disabled}
        type="button"
        onClick={() => inputRef.current?.click()}
      >
        Browse Files
      </Button>
      <input
        ref={inputRef}
        type="file"
        multiple
        accept={accept}
        onChange={handleFileInput}
        className="hidden"
        disabled={disabled}
      />
    </div>
  );
}
