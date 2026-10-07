"""Demo corpus with labeled relevance judgments."""

# Each doc: id + text. QUERIES maps query -> list of relevant doc ids.
DOCUMENTS = [
    {"id": "d1", "text": "Enterprise contracts include a 30-day cancellation window from the signature date. Refunds are pro-rated."},
    {"id": "d2", "text": "Standard plans are billed monthly and are non-refundable after the first 7 days."},
    {"id": "d3", "text": "To request a refund, email billing@acme.com with your contract ID and reason."},
    {"id": "d4", "text": "SSO via SAML is available on enterprise plans only. Standard plans use email/password auth."},
    {"id": "d5", "text": "The API rate limit is 1000 requests per minute per key on enterprise, 100 on standard."},
    {"id": "d6", "text": "Data is retained for 13 months after contract termination, then permanently purged."},
    {"id": "d7", "text": "Enterprise onboarding includes a dedicated solutions engineer for the first 90 days."},
    {"id": "d8", "text": "SLA credits are issued automatically when monthly uptime falls below 99.9%."},
    {"id": "d9", "text": "The platform runs in us-east-1, eu-west-1, and ap-southeast-2 regions."},
    {"id": "d10", "text": "Webhook deliveries retry with exponential backoff for up to 24 hours."},
    {"id": "d11", "text": "Audit logs are available for 90 days on enterprise plans via the compliance API."},
    {"id": "d12", "text": "Password requirements: 12+ characters, mixed case, numbers, and symbols."},
]

QUERIES = {
    "refund policy enterprise": ["d1", "d3"],
    "how do I get my money back": ["d1", "d3"],
    "SSO setup enterprise contract 30-day": ["d4", "d1"],
    "is single sign on available": ["d4"],
    "api limits per key": ["d5"],
    "how fast can I call the api": ["d5"],
    "data deletion after contract ends": ["d6"],
    "uptime guarantee credits": ["d8"],
}
