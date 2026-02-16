import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "FineTuneFlow",
  description:
    "Local fine-tuning pipeline: docs → dataset → LoRA/QLoRA SFT → export",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-background antialiased">
        <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
          <div className="container flex h-14 items-center px-6">
            <span className="text-lg font-bold tracking-tight">
              FineTuneFlow
            </span>
          </div>
        </header>
        <main className="container mx-auto px-6 py-8">{children}</main>
      </body>
    </html>
  );
}
