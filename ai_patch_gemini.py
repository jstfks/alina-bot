async def _call_gemini(
    messages: list[dict],
    max_tokens: int,
    temperature: float,
) -> tuple[Optional[str], Optional[str], int, int, int]:
    """
    Returns: (text, model_name, prompt_tokens, completion_tokens, total_tokens)
    Returns (None, None, 0, 0, 0) on failure.
    """
    if not GEMINI_API_KEY:
        return None, None, 0, 0, 0

    system_prompt = ""
    gemini_messages: list[dict] = []
    for msg in messages:
        role = msg["role"]
        if role == "system":
            system_prompt = msg["content"]
        elif role == "user":
            gemini_messages.append({"role": "user", "parts": [{"text": msg["content"]}]})
        elif role == "assistant":
            gemini_messages.append({"role": "model", "parts": [{"text": msg["content"]}]})

    # Gemini требует строгого чередования user/model
    deduped: list[dict] = []
    for m in gemini_messages:
        if deduped and deduped[-1]["role"] == m["role"]:
            deduped[-1]["parts"][0]["text"] += "\n" + m["parts"][0]["text"]
        else:
            deduped.append(m)

    if not deduped or deduped[0]["role"] != "user":
        return None, None, 0, 0, 0

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent"
    )
    session = await get_http_session()
    try:
        async with session.post(
            url,
            headers={
                "x-goog-api-key": GEMINI_API_KEY,
                "Content-Type": "application/json",
            },
            json={
                "system_instruction": {"parts": [{"text": system_prompt}]},
                "contents": deduped,
                "generationConfig": {
                    "maxOutputTokens": max_tokens,
                    "temperature": temperature,
                },
            },
            timeout=aiohttp.ClientTimeout(total=25),
        ) as resp:
            if _is_rate_limited(resp.status):
                log.warning("[Gemini] rate-limited (%s)", resp.status)
                return None, None, 0, 0, 0
            data = await resp.json()
            if "candidates" not in data:
                log.warning("[Gemini] неожиданный ответ: %s", data.get("error"))
                return None, None, 0, 0, 0
            text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            
            # Extract token usage from Gemini response
            usage = data.get("usageMetadata", {})
            prompt_tokens = usage.get("promptTokenCount", 0)
            completion_tokens = usage.get("candidatesTokenCount", 0)
            total_tokens = usage.get("totalTokenCount", 0)
            
            log.info("[Gemini] успех: %s (tokens: %d)", GEMINI_MODEL, total_tokens)
            return text, GEMINI_MODEL, prompt_tokens, completion_tokens, total_tokens
    except asyncio.TimeoutError:
        log.warning("[Gemini] timeout")
        return None, None, 0, 0, 0
    except Exception as exc:
        log.warning("[Gemini] исключение: %s", exc)
        return None, None, 0, 0, 0