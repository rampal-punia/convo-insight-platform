import './globals.css';

export const metadata = {
  title: 'ConvoInsight',
  description: 'Customer conversational intelligence platform',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-slate-50 text-slate-900 antialiased">{children}</body>
    </html>
  );
}
