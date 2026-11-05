"""Browser regression against a running GPU server; needs optional Playwright.

No model runs in this process. The server must be configured for the GPU.
"""
import argparse
import json
import os
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default='http://127.0.0.1:3218/')
    ap.add_argument('--resolve-host', help='Optional Chromium host-resolver rule for deployment checks')
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    with sync_playwright() as driver:
        options = ['--no-sandbox', '--no-proxy-server']
        if args.resolve_host:
            options.append('--host-resolver-rules=' + args.resolve_host)
        browser = driver.chromium.launch(executable_path=os.getenv('QOT_CHROMIUM'), headless=True, args=options)
        try:
            page = browser.new_page(viewport={'width': 1440, 'height': 1000})
            errors, posts = [], []
            page.on('pageerror', lambda exc: errors.append(str(exc)))
            page.on('request', lambda req: posts.append(req.url) if req.method == 'POST' else None)
            page.goto(args.url)
            page.wait_for_function('() => connected && state?.ready && !state.busy', timeout=60000)
            assert page.evaluate('state.score_curve') == 'smoothstep'
            assert page.evaluate('state.max_thinking_tokens') == 131072
            assert page.evaluate('state.total_context_tokens') == 262144
            assert page.locator('#override').get_attribute('max') == '131072'
            assert 'RTX 3090' in page.locator('#gpu-label').inner_text()
            page.locator('#prompt').fill('Say hello.')
            page.locator('#answer-limit').fill('32')
            page.locator('#primary').click()  # ONE click; no decision request.
            page.wait_for_function('() => !run && lastResult?.phase === "complete"', timeout=60000)
            first = page.evaluate('lastResult')
            assert len(posts) == 1 and posts[0].endswith('/api/generate'), posts
            assert first['prefill_count'] == 1 and first['cache_continuous']
            assert 0 <= first['score'] <= 1 and first['budget'] == first['predicted_budget']
            assert first['budget'] == int(first['score'] ** 2 * (3 - 2 * first['score']) * 131072 + .5)
            assert 'probabilities' not in first and 'decision' not in first
            assert page.locator('#probabilities').count() == 0
            point = page.locator('#score-point')
            assert abs(float(point.get_attribute('cx')) - (4 + 292 * first['score'])) < 1e-4
            assert abs(float(point.get_attribute('cy')) - (76 - 72 * first['score'] ** 2 * (3 - 2 * first['score']))) < 1e-4
            assert page.locator('#model-score').inner_text() == f"{first['score']:.6f}"
            assert first['answer_tokens'] > 0 and page.locator('.answer').last.inner_text().strip()
            assert 'same inference cache' in page.locator('.decision-note').last.inner_text()
            assert 'Send prompt' in page.locator('#primary').inner_text()
            assert not page.locator('#live-dot').evaluate("el => el.classList.contains('active')")
            page.screenshot(path=str(root / 'artifacts/one-two-browser-desktop.png'), full_page=True)
            # A second independent prompt still takes exactly one submission.
            page.wait_for_function('() => !state.busy && !document.querySelector("#primary").disabled')
            page.locator('#prompt').fill("Mira's application was approved. Is the application approved? Answer yes or no.")
            page.locator('#answer-limit').fill('256')
            page.locator('#override').fill('')
            page.locator('#primary').click()
            page.wait_for_function('() => !run && lastResult?.phase === "complete" && lastResult?.text.startsWith("Mira")', timeout=60000)
            second = page.evaluate('lastResult')
            assert len(posts) == 2 and all(p.endswith('/api/generate') for p in posts), posts
            assert second['budget'] == second['predicted_budget'] and second['predicted_budget'] > 0
            assert second['predicted_budget'] == int(second['score'] ** 2 * (3 - 2 * second['score']) * 131072 + .5)
            assert second['predicted_budget'] not in (0,128,512,2048)
            assert second['prefill_count'] == 1 and second['cache_continuous']
            assert second['thinking_tokens'] > 0 and second['answer_tokens'] > 0
            assert any(c['status']=='complete' for c in second['decision_calls']), second['decision_calls']
            assert page.locator('#decision-calls').inner_text().strip() != 'No decisions yet.'
            assert page.locator('.answer').count() == 2
            assert all(page.locator('.answer').nth(i).inner_text().strip() for i in range(2))
            page.set_viewport_size({'width': 390, 'height': 844})
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            page.screenshot(path=str(root / 'artifacts/one-two-browser-mobile.png'), full_page=True)
            assert not errors, errors
            report = {'url': args.url, 'post_requests': posts, 'javascript_errors': errors,
                      'one_click_per_prompt': True, 'separate_decision_requests': 0,
                      'first_normal_answer': first['answer'], 'positive_budget_normal_answer': second['answer'],
                      'prefills_per_run': [first['prefill_count'], second['prefill_count']],
                      'score_curve': 'smoothstep',
                      'maximum_thinking_tokens': 131072,
                      'scores': [first['score'], second['score']],
                      'mapped_budgets': [first['predicted_budget'], second['predicted_budget']],
                      'effective_budgets': [first['budget'], second['budget']],
                      'same_cache_each_run': True, 'mobile_no_overflow': True,
                      'answer_endings': [first['answer_end'], second['answer_end']]}
            (root / 'artifacts/one-two-browser.json').write_text(json.dumps(report, indent=2))
            print(json.dumps(report, indent=2), flush=True)
        finally:
            browser.close()


if __name__ == '__main__':
    main()
