export const config = { runtime: "edge" };

function jsonResponse(body, status) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

export default async function handler(req) {
  if (req.method !== "POST") {
    return jsonResponse({ error: "Method not allowed" }, 405);
  }

  const apiKey = process.env.NVIDIA_API_KEY;
  if (!apiKey) {
    return jsonResponse(
      { error: "NVIDIA_API_KEY is not configured on the server." },
      500
    );
  }

  let body;
  try {
    body = await req.json();
  } catch {
    return jsonResponse({ error: "Invalid JSON body" }, 400);
  }

  if (
    typeof body.system !== "string" ||
    !Array.isArray(body.messages) ||
    body.messages.some(
      (message) =>
        !message ||
        !["user", "assistant"].includes(message.role) ||
        typeof message.content !== "string"
    )
  ) {
    return jsonResponse({ error: "Invalid chat request" }, 400);
  }

  let upstream;
  try {
    upstream = await fetch("https://integrate.api.nvidia.com/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: process.env.NVIDIA_MODEL || "meta/llama-3.3-70b-instruct",
        max_tokens: 1000,
        messages: [{ role: "system", content: body.system }, ...body.messages],
        stream: true,
      }),
    });
  } catch (error) {
    console.error("NVIDIA API request failed:", error);
    return jsonResponse({ error: "Unable to connect to the NVIDIA API." }, 502);
  }

  return new Response(upstream.body, {
    status: upstream.status,
    headers: {
      "Content-Type": upstream.headers.get("Content-Type") ?? "application/json",
    },
  });
}
