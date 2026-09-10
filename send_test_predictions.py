# STAGE 6 (Utilization)
# Modified: each sample email now carries its known true label, so this
# script can tell you whether the live API got it right — not just what
# it predicted. This is separate from Stage 4's real accuracy numbers
# (Evaluation_Report.md), which are measured on thousands of held-out
# emails, not these 30 illustrative examples.

import argparse
import time

import requests

API_URL = "http://localhost:8000/predict"

# (text, true_label) — 1 = phishing, 0 = safe
SAMPLE_EMAILS = [
    # phishing
    ("URGENT: Your account has been suspended. Verify your password immediately at http://secure-login-update.com or lose access within 24 hours.", 1),
    ("Congratulations! You have won a free prize. Click here to claim: http://win-now-prize.net", 1),
    ("Dear customer, we detected unusual activity on your bank account. Confirm your identity here: http://bank-verify-secure.com/login", 1),
    ("Your payment failed. Update your card details now to avoid service interruption: http://billing-update-portal.com", 1),
    ("Action required: your mailbox is full. Verify your credentials to restore access http://mail-quota-fix.net", 1),
    ("You have an unclaimed refund of 450 EUR waiting. Submit your account details to receive payment.", 1),
    ("Security alert: someone signed in to your account from a new device. If this was not you, reset your password here http://account-secure-reset.com", 1),
    ("Final notice: your subscription expires today. Renew immediately using this link to keep your benefits.", 1),
    ("IMPORTANT - Your package could not be delivered. Pay the customs fee here to reschedule: http://parcel-delivery-fee.com", 1),
    ("Limited offer! Get 90% discount on all products. Offer valid for the next 2 hours only. Click to shop now.", 1),

    # safe
    ("Hi team, attached are the meeting notes from Tuesday. Let me know if I missed anything. Best, Anna", 0),
    ("Hello, just confirming our appointment for Thursday at 3pm. Looking forward to it.", 0),
    ("Please find the quarterly report attached. Happy to walk through the numbers if useful.", 0),
    ("Reminder: the office will be closed next Monday for the public holiday.", 0),
    ("Thanks for sending the draft. I have a few small comments, will share them by Friday.", 0),
    ("The build passed on the main branch. Deployment is scheduled for tomorrow morning.", 0),
    ("Hi, could you send me the updated slide deck when you get a chance? No rush.", 0),
    ("Following up on my previous message about the workshop registration. Let me know if you need anything from me.", 0),
    ("Good morning, the client rescheduled to next week. I have updated the shared calendar.", 0),
    ("Lunch is at 12:30 in the usual place if you want to join.", 0),

    # short / mixed
    ("Verify your account now.", 1),
    ("Meeting moved to 4pm.", 0),
    ("Click here to reset password immediately http://reset-now-secure.com", 1),
    ("Thanks, received.", 0),
    ("Your invoice is attached for review.", 0),
    ("Free gift card waiting for you, claim within 1 hour!", 1),
    ("Can we push the call by 15 minutes?", 0),
    ("Suspicious login detected, confirm your details here.", 1),
    ("Report submitted, no further action needed.", 0),
    ("Update your billing information to continue your service without interruption.", 1),
]


def send_one(text):
    try:
        response = requests.post(API_URL, json={"text": text}, timeout=10)
        if response.status_code == 200:
            return response.json()
        print(f"  HTTP {response.status_code}: {response.text[:80]}")
        return None
    except requests.exceptions.ConnectionError:
        print("ERROR: cannot reach the API. Is 05_deploy.py running?")
        print("  Start it with: uvicorn 05_deploy:app --host 0.0.0.0 --port 8000")
        return "STOP"


def main():
    parser = argparse.ArgumentParser(description="Send sample emails to the API.")
    parser.add_argument("--repeat", type=int, default=1,
                        help="how many times to go through the sample list")
    args = parser.parse_args()

    print("STAGE 6 — CREATE VALUE (simulated requests)")
    print("=" * 60)
    print(f"Sending {len(SAMPLE_EMAILS)} emails x {args.repeat} round(s) "
          f"to {API_URL}\n")

    sent = 0
    correct = 0
    phishing_count = 0
    false_positives = []  # true safe, predicted phishing
    false_negatives = []  # true phishing, predicted safe

    for round_number in range(args.repeat):
        for text, true_label in SAMPLE_EMAILS:
            result = send_one(text)
            if result == "STOP":
                break
            if result is None:
                continue

            sent += 1
            pred_label = result["label"]
            is_correct = (pred_label == true_label)
            correct += int(is_correct)
            if pred_label == 1:
                phishing_count += 1

            if not is_correct:
                if true_label == 0:
                    false_positives.append(text)
                else:
                    false_negatives.append(text)

            mark = "✓" if is_correct else "✗ WRONG"
            preview = text[:40] + ("..." if len(text) > 40 else "")
            score = result.get("decision_score")
            score_text = f"{score:.3f}" if score is not None else "n/a"
            print(f"  [{mark:8}] pred={result['prediction']:15} true={'phishing' if true_label else 'safe':8} "
                  f"score={score_text:>7}  {preview}")

            time.sleep(0.05)

    print("\n" + "=" * 60)
    print(f"Sent: {sent} predictions")
    if sent:
        print(f"Correct: {correct}/{sent} ({correct/sent*100:.1f}%)")
        print(f"Flagged as phishing: {phishing_count} ({phishing_count/sent*100:.1f}%)")
    if false_positives:
        print(f"\nFalse positives (safe email flagged as phishing) — {len(false_positives)}:")
        for t in false_positives:
            print(f"  - {t[:70]}")
    if false_negatives:
        print(f"\nFalse negatives (phishing email missed) — {len(false_negatives)}:")
        for t in false_negatives:
            print(f"  - {t[:70]}")

    print("\nNote: this is a sanity check on 30 hand-picked examples, not a "
          "real accuracy measurement — see Evaluation_Report.md for that.")
    print("\nNext: python 06_monitor.py")


if __name__ == "__main__":
    main()