# === main.py ===============================================================
from __future__ import annotations

import argparse
from datetime import datetime

from simulation_engine import simulate_query_handling
from base_agents import run_groupchat_flow
from metrics import summarize_logs
from synthetic_data import generate_queries
from memory import log_interaction
from voice import synthesize

def _batch_demo(use_router: bool, use_llm_router: bool, session_id: str | None, voice: bool,
                tts_cmd: str | None, tts_model: str | None, tts_voice: str | None) -> None:
    print("\n🚀  Running autonomous GroupChat batch …\n")
    questions = [
        "Why was I charged a $15 fee?",
        "Can I cancel a pending transfer?",
        "What is a margin call?",
        "How do I dispute a charge?",
    ]
    for i, q in enumerate(questions, 1):
        print(f"\n== Query {i} ==\n")
        response = run_groupchat_flow(q, session_id=session_id)
        if voice:
            audio_path = synthesize(
                response,
                filename_stem=f"{session_id or 'batch'}_{i}",
                model=tts_model,
                voice=tts_voice,
                command=tts_cmd,
            )
            log_interaction(
                agent_name="Voice",
                query=q,
                response=response,
                metadata={"session_id": session_id, "tts_path": str(audio_path)},
            )
    print("\n✅  GroupChat simulation complete!\n")

def _interactive_cli(use_router: bool, use_llm_router: bool, session_id: str | None, voice: bool,
                     tts_cmd: str | None, tts_model: str | None, tts_voice: str | None) -> None:
    print("\n🗣️  Enter support queries (type 'exit' to quit)\n")
    qid = 1
    while True:
        text = input("🧑‍💻 You: ").strip()
        if text.lower() in {"exit", "quit"}:
            print("👋  Bye!")
            break
        response = simulate_query_handling(
            qid,
            text,
            session_id=session_id,
            use_router=use_router,
            use_llm_router=use_llm_router,
        )
        if voice:
            audio_path = synthesize(
                response,
                filename_stem=f"{session_id or 'cli'}_{qid}",
                model=tts_model,
                voice=tts_voice,
                command=tts_cmd,
            )
            log_interaction(
                agent_name="Voice",
                query=text,
                response=response,
                metadata={"session_id": session_id, "tts_path": str(audio_path)},
            )
        qid += 1


def _synthetic_demo(count: int, use_router: bool, use_llm_router: bool, session_id: str | None,
                    voice: bool, tts_cmd: str | None, tts_model: str | None,
                    tts_voice: str | None) -> None:
    print("\n🧪  Running synthetic batch …\n")
    questions = generate_queries(count=count)
    for i, q in enumerate(questions, 1):
        response = simulate_query_handling(
            i,
            q,
            session_id=session_id,
            use_router=use_router,
            use_llm_router=use_llm_router,
        )
        if voice:
            audio_path = synthesize(
                response,
                filename_stem=f"{session_id or 'synthetic'}_{i}",
                model=tts_model,
                voice=tts_voice,
                command=tts_cmd,
            )
            log_interaction(
                agent_name="Voice",
                query=q,
                response=response,
                metadata={"session_id": session_id, "tts_path": str(audio_path)},
            )
    print("\n✅  Synthetic simulation complete!\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Customer Support Simulator")
    parser.add_argument("--mode", choices=["cli", "batch", "synthetic"], default="cli")
    parser.add_argument("--router", action="store_true", help="Enable router-based specialist flow")
    parser.add_argument("--llm-router", action="store_true", help="Use LLM router instead of rules")
    parser.add_argument("--session-id", default=None, help="Session identifier for logs")
    parser.add_argument("--voice", action="store_true", help="Enable TTS output via Chatterbox")
    parser.add_argument("--tts-cmd", default=None, help="CLI command template for Chatterbox")
    parser.add_argument("--tts-model", default=None, help="Model name for Chatterbox")
    parser.add_argument("--tts-voice", default=None, help="Voice name for Chatterbox")
    parser.add_argument("--synthetic-count", type=int, default=8)
    parser.add_argument("--metrics", action="store_true", help="Print metrics summary after run")
    args = parser.parse_args()

    session_id = args.session_id or datetime.utcnow().strftime("%Y%m%d%H%M%S")

    print("Customer Support Simulator")
    if args.mode == "batch":
        _batch_demo(args.router, args.llm_router, session_id, args.voice,
                    args.tts_cmd, args.tts_model, args.tts_voice)
    elif args.mode == "synthetic":
        _synthetic_demo(args.synthetic_count, args.router, args.llm_router, session_id,
                        args.voice, args.tts_cmd, args.tts_model, args.tts_voice)
    else:
        _interactive_cli(args.router, args.llm_router, session_id, args.voice,
                         args.tts_cmd, args.tts_model, args.tts_voice)

    if args.metrics:
        summary = summarize_logs(session_id=session_id)
        print("\n📊 Metrics Summary")
        for key, value in summary.items():
            print(f"- {key}: {value}")
