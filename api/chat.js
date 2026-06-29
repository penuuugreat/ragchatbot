
export const config = { runtime: "edge" };

const OPENAI_API_KEY = "https://api.openai.com/v1/chat/completions";

export default async function handler(req) {
  // Only allow POST
  if (req.method !== "POST") {
    return new Response(JSON.stringify({ error: "Method not allowed" }), {
      status: 405,
      headers: { "Content-Type": "application/json" },
    });
  }

  const apiKey = process.env.OPENAI_API_KEY;
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
  const upstream = await fetch("https://api.openai.com/v1/chat/completions", {
  headers: {
    "Authorization": `Bearer ${apiKey}`,
  },
  body: JSON.stringify({
    model: "gpt-4o-mini",  // cheap, fast — good for portfolio
    max_tokens: 1000,
    messages: [
      { role: "system", content: body.system },
      ...body.messages,
    ],
    stream: true,
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

