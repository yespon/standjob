import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "岗标 AI 教练 Demo",
  description: "基于 AI 的智能辅导",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN" className="h-full">
      <body className="h-full">{children}</body>
    </html>
  );
}
