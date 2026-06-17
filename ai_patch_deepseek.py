async def _call_deepseek(
    messages: list[dict],
    max_tokens: int,
    temperature: float,
) -> tuple[Optional[str], Optional[str], int, int, int]:
    """
    Returns: (text, model_name, prompt_tokens, completion_tokens, total_tokens)
    Returns (None, None, 0, 0, 0) on failure.
    """
    if not DEEPSEEK_API_KEY:
        return None, None, 0, 0, 0
    session = await get_http_session()
    try:
        async with session.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": DEEPSEEK_MODEL,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            if _is_rate_limited(resp.status):
                log.warning("[DeepSeek] rate-limited (%s)", resp.status)
                return None, None, 0, 0, 0
            data = await resp.json()
            if "choices" not in data:
                err = data.get("error", {})
                if "Insufficient Balance" in str(err):
                    log.error("[DeepSeek] БАЛАНС КОНЧИЛСЯ — пополните счёт на platform.deepseek.com")
                else:
                    log.warning("[DeepSeek] неожиданный ответ: %s", err)
                return None, None, 0, 0, 0
            text = data["choices"][0]["message"]["content"].strip()
            
            # Extract token usage from DeepSeek response
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
            
            log.info("[DeepSeek] успех: %s (tokens: %d)", DEEPSEEK_MODEL, total_tokens)
            return text, DEEPSEEK_MODEL, prompt_tokens, completion_tokens, total_tokens
    except asyncio.TimeoutError:
        log.warning("[DeepSeek] timeout")
        return None, None, 0, 0, 0
    except Exception as exc:
        log.warning("[DeepSeek] исключение: %s", exc)
        return None, None, 0, 0, 0