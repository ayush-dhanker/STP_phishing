# STAGE 6 (Utilization)

import argparse
import time

import requests

API_URL = "http://localhost:8000/predict"
 
SAMPLE_EMAILS = [
    #phishing
    "URGENT: Your account has been suspended. Verify your password immediately at http://secure-login-update.com or lose access within 24 hours.",
    "Congratulations! You have won a free prize. Click here to claim: http://win-now-prize.net",
    "Dear customer, we detected unusual activity on your bank account. Confirm your identity here: http://bank-verify-secure.com/login",
    "Your payment failed. Update your card details now to avoid service interruption: http://billing-update-portal.com",
    "Action required: your mailbox is full. Verify your credentials to restore access http://mail-quota-fix.net",
    "You have an unclaimed refund of 450 EUR waiting. Submit your account details to receive payment.",
    "Security alert: someone signed in to your account from a new device. If this was not you, reset your password here http://account-secure-reset.com",
    "Final notice: your subscription expires today. Renew immediately using this link to keep your benefits.",
    "IMPORTANT - Your package could not be delivered. Pay the customs fee here to reschedule: http://parcel-delivery-fee.com",
    "Limited offer! Get 90% discount on all products. Offer valid for the next 2 hours only. Click to shop now.",

    #safe
    "Hi team, attached are the meeting notes from Tuesday. Let me know if I missed anything. Best, Anna",
    "Hello, just confirming our appointment for Thursday at 3pm. Looking forward to it.",
    "Please find the quarterly report attached. Happy to walk through the numbers if useful.",
    "Reminder: the office will be closed next Monday for the public holiday.",
    "Thanks for sending the draft. I have a few small comments, will share them by Friday.",
    "The build passed on the main branch. Deployment is scheduled for tomorrow morning.",
    "Hi, could you send me the updated slide deck when you get a chance? No rush.",
    "Following up on my previous message about the workshop registration. Let me know if you need anything from me.",
    "Good morning, the client rescheduled to next week. I have updated the shared calendar.",
    "Lunch is at 12:30 in the usual place if you want to join.",


    "Verify your account now.",
    "Meeting moved to 4pm.",
    "Click here to reset password immediately http://reset-now-secure.com",
    "Thanks, received.",
    "Your invoice is attached for review.",
    "Free gift card waiting for you, claim within 1 hour!",
    "Can we push the call by 15 minutes?",
    "Suspicious login detected, confirm your details here.",
    "Report submitted, no further action needed.",
    "Update your billing information to continue your service without interruption.",
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
    phishing_count = 0

    for round_number in range(args.repeat):
        for text in SAMPLE_EMAILS:
            result = send_one(text)
            if result == "STOP":
                return
            if result is None:
                continue

            sent += 1
            if result["label"] == 1:
                phishing_count += 1

            preview = text[:45] + ("..." if len(text) > 45 else "")
            score = result.get("decision_score")
            score_text = f"{score:.3f}" if score is not None else "n/a"
            print(f"  [{result['prediction']:15}] score={score_text:>7}  {preview}")

            time.sleep(0.05)

    print("\n" + "=" * 60)
    print(f"Sent: {sent} predictions")
    if sent:
        print(f"Flagged as phishing: {phishing_count} ({phishing_count/sent*100:.1f}%)")
    print("\nNext: python 06_monitor.py")


if __name__ == "__main__":
    main()