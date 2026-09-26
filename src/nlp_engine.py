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


def predict_complaint_category(text):
    """
    Wrapper function to maintain compatibility with app.py imports.
    Returns only the predicted category string.
    """
    category, _ = classify_complaint_narrative(text)
    return category


def generate_dispute_letter(narrative, company_name="Financial Institution", consumer_name="Consumer", county="Regional"):
    """
    Generates a structured CFPB dispute letter template matching app.py inputs.
    """
    category, relief_rate = classify_complaint_narrative(narrative)
    
    return f"""FORMAL NOTICE OF DISPUTE & REQUEST FOR INVESTIGATION

TO: Compliance Department
    {company_name}

FROM: {consumer_name}
LOCATION: {county} County
DATE: September 26, 2026
RE: Formal Consumer Grievance - {category}

To Whom It May Concern,

I am writing to formally dispute action(s) taken regarding my mortgage account/application with {company_name}. 

Dispute Category: {category}
Historical Category Relief Rate: {relief_rate * 100:.0f}%

STATEMENT OF FACTS:
{narrative}

REQUESTED ACTION:
Pursuant to the Consumer Financial Protection Act and applicable federal consumer financial laws (including the Fair Credit Reporting Act and Equal Credit Opportunity Act), I hereby request an immediate review, investigation into potential systemic lending disparities, and full resolution/correction of improper charges or decisions.

Sincerely,

{consumer_name}
"""