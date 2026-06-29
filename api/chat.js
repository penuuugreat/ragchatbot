/**
 * api/chat.js  —  Vercel Edge Function
 *
 * Proxies requests to the Anthropic Messages API so the API key never
 * reaches the browser.  Set ANTHROPIC_API_KEY in your Vercel project's
 * Environment Variables (not in .env.local — that file is client-side
 * when used with VITE_* prefix).
 *
 * Usage from the frontend:
 *   POST /api/chat   { model, max_tokens, system, messages, stream }
 */

export const config = { runtime: "edge" };

const ANTHROPIC_API = "https://api.anthropic.com/v1/messages";

export default async function handler(req) {
  // Only allow POST
  if (req.method !== "POST") {
    return new Response(JSON.stringify({ error: "Method not allowed" }), {
      status: 405,
      headers: { "Content-Type": "application/json" },
    });
  }

  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    return new Response(
      JSON.stringify({ error: "ANTHROPIC_API_KEY is not configured on the server." }),
      { status: 500, headers: { "Content-Type": "application/json" } }
    );
  }

  let body;
  try {
    body = await req.json();
  } catch {
    return new Response(JSON.stringify({ error: "Invalid JSON body" }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }

  // Forward to Anthropic, streaming if requested
  const upstream = await fetch(ANTHROPIC_API, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": apiKey,
      "anthropic-version": "2023-06-01",
    },
    body: JSON.stringify({
      model: body.model ?? "claude-sonnet-4-6",
      max_tokens: body.max_tokens ?? 1000,
      system: body.system,
      messages: body.messages,
      stream: body.stream ?? false,
    }),
  });

  // Stream the response straight back to the browser
  return new Response(upstream.body, {
    status: upstream.status,
    headers: {
      "Content-Type": upstream.headers.get("Content-Type") ?? "application/json",
      "Access-Control-Allow-Origin": "*",
    },
  });
}

