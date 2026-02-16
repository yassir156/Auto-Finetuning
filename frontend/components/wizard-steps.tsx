"use client";

import { cn } from "@/lib/utils";
import { WIZARD_STEPS, WizardStep } from "@/lib/store";

const STEP_LABELS: Record<WizardStep, string> = {
  model: "Model",
  task: "Task",
  data: "Data",
  preview: "Preview",
  review: "Review",
  hardware: "Hardware",
  train: "Train",
  export: "Export",
};

interface WizardStepsProps {
  currentStep: WizardStep;
  onStepClick?: (step: WizardStep) => void;
  completedSteps?: WizardStep[];
}

export function WizardSteps({
  currentStep,
  onStepClick,
  completedSteps = [],
}: WizardStepsProps) {
  const currentIdx = WIZARD_STEPS.indexOf(currentStep);

  return (
    <nav className="mb-8">
      <ol className="flex items-center gap-2">
        {WIZARD_STEPS.map((step, idx) => {
          const isActive = step === currentStep;
          const isCompleted = completedSteps.includes(step) || idx < currentIdx;
          const isClickable = onStepClick && (isCompleted || idx <= currentIdx);

          return (
            <li key={step} className="flex items-center gap-2">
              {idx > 0 && (
                <div
                  className={cn(
                    "h-px w-6 sm:w-10",
                    isCompleted ? "bg-primary" : "bg-border"
                  )}
                />
              )}
              <button
                onClick={() => isClickable && onStepClick?.(step)}
                disabled={!isClickable}
                className={cn(
                  "flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-medium transition-colors",
                  isActive && "bg-primary text-primary-foreground",
                  isCompleted && !isActive && "bg-primary/10 text-primary",
                  !isActive && !isCompleted && "bg-muted text-muted-foreground",
                  isClickable && "cursor-pointer hover:opacity-80",
                  !isClickable && "cursor-default"
                )}
              >
                <span
                  className={cn(
                    "flex h-5 w-5 items-center justify-center rounded-full text-[10px] font-bold",
                    isActive && "bg-primary-foreground text-primary",
                    isCompleted && !isActive && "bg-primary text-primary-foreground",
                    !isActive && !isCompleted && "bg-muted-foreground/20 text-muted-foreground"
                  )}
                >
                  {isCompleted && !isActive ? "✓" : idx + 1}
                </span>
                <span className="hidden sm:inline">{STEP_LABELS[step]}</span>
              </button>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
