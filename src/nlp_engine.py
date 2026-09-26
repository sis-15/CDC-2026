import re

def classify_complaint_narrative(text):
    """
    Classifies consumer narrative text into sub-issues and maps historical relief rates.
    (Can be upgraded with TF-IDF / Scikit-Learn models).
    """
    text_lower = text.lower()
    
    if any(k in text_lower for k in ["fee", "charge", "cost", "closing"]):
        category = "Disclosure & Fee Violations"
        relief_rate = 0.64
    elif any(k in text_lower for k in ["discrim", "race", "denial", "redlin"]):
        category = "Equal Credit Opportunity / Redlining"
        relief_rate = 0.42
    elif any(k in text_lower for k in ["modify", "payment", "escrow", "servicing"]):
        category = "Loan Servicing & Payment Handling"
        relief_rate = 0.55
    else:
        category = "General Mortgage Dispute"
        relief_rate = 0.50
        
    return category, relief_rate

def generate_dispute_letter(county, narrative, category):
    """Generates a structured CFPB dispute letter template."""
    return f"""[Date]
To: Consumer Financial Protection Bureau / Mortgage Servicer
Subject: Formal Dispute Regarding {category}

I am submitting a formal dispute regarding a mortgage lending/servicing issue in {county} County.

Description of Issue:
{narrative}

Requested Action:
Immediate review, investigation into systemic lending disparities, and full resolution of improper charges or decisions.
"""