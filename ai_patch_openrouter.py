async def _call_openrouter(
    messages: list[dict],
    max_tokens: int,
    temperature: float,
) -> tuple[Optional[str], Optional[str], int, int, int]:
    """
    Returns: (text, model_name, prompt_tokens, completion_tokens, total_tokens)
    Returns (None, None, 0, 0, 0) on failure.
    """
    if not OPENROUTER_API_KEY:
        return None, None, 0, 0, 0
    session = await get_http_session()
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/alina-bot",
        "X-Title": "Alina Bot",
    }
    for model in OPENROUTER_FALLBACK_MODELS:
        try:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": 0.92,
                },
                timeout=aiohttp.ClientTimeout(total=35),
            ) as resp:
                if _is_rate_limited(resp.status):
                    log.warning("[OpenRouter] %s rate-limited (%s)", model, resp.status)
                    await asyncio.sleep(0.3)
                    continue
                data = await resp.json()
                if "choices" not in data:
                    err  = data.get("error", {})
                    code = err.get("code", resp.status)
                    msg  = str(err.get("message", ""))[:120]
                    if resp.status == 404:
                        log.warning("[OpenRouter] %s — модель не найдена (снята?): %s", model, msg)
                    else:
                        log.warning("[OpenRouter] %s — ошибка %s: %s", model, code, msg)
                    await asyncio.sleep(0.3)
                    continue
                text = data["choices"][0]["message"]["content"].strip()
                text = _strip_think_tags(text)
                if text:
                    # Extract token usage from response
                    usage = data.get("usage", {})
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    total_tokens = usage.get("total_tokens", 0)
                    log.info("[OpenRouter] успех: %s (tokens: %d)", model, total_tokens)
                    return text, model, prompt_tokens, completion_tokens, total_tokens
                log.warning("[OpenRouter] %s — пустой ответ после обрезки think-блоков", model)
                await asyncio.sleep(0.3)
                continue
        except asyncio.TimeoutError:
            log.warning("[OpenRouter] %s timeout", model)
            await asyncio.sleep(0.3)
        except Exception as exc:
            log.warning("[OpenRouter] %s исключение: %s", model, exc)
            await asyncio.sleep(0.3)

    log.error("[OpenRouter] все модели недоступны")
    return None, None, 0, 0, 0