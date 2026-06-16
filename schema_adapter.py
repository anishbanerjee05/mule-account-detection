import yaml
import json
import os
from datetime import datetime

class SchemaAdapter:
    """
    Bank-agnostic transaction schema adapter.
    Translates any bank's transaction format into the standard
    mule detection schema using a YAML configuration file.
    
    Adding a new bank requires only a new YAML config file.
    Zero code changes to the detection pipeline.
    """

    STANDARD_FIELDS = [
        'account_from',    # Sender account ID
        'account_to',      # Receiver account ID
        'bank_from',       # Sender bank code
        'bank_to',         # Receiver bank code
        'amount_paid',     # Transaction amount
        'currency',        # Currency code
        'payment_format',  # Transaction type (UPI, NEFT, IMPS, RTGS etc.)
        'timestamp',       # Transaction datetime
        'transaction_id',  # Unique transaction reference
        'is_laundering',   # Ground truth label (0 in production)
    ]

    def __init__(self, config_path: str):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Schema config not found: {config_path}")

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.bank_name     = self.config.get('bank_name', 'Unknown')
        self.field_map     = self.config.get('field_mappings', {})
        self.defaults      = self.config.get('defaults', {})
        self.transformers  = self.config.get('transformers', {})

        print(f"Schema adapter loaded for: {self.bank_name}")
        print(f"Field mappings: {len(self.field_map)}")

    def translate(self, raw_event: dict) -> dict:
        """
        Translate a bank-specific transaction event to the standard schema.
        Returns a standardised event dict ready for the Kafka producer.
        """
        standard = {}

        for std_field in self.STANDARD_FIELDS:
            # Check if there is a mapping for this field
            bank_field = self.field_map.get(std_field)

            if bank_field and bank_field in raw_event:
                value = raw_event[bank_field]

                # Apply transformer if defined
                transformer = self.transformers.get(std_field)
                if transformer == 'upper':
                    value = str(value).upper()
                elif transformer == 'lower':
                    value = str(value).lower()
                elif transformer == 'float':
                    value = float(value)
                elif transformer == 'int':
                    value = int(value)
                elif transformer == 'str':
                    value = str(value)
                elif transformer == 'timestamp_iso':
                    # Convert various date formats to ISO
                    try:
                        from dateutil import parser as dateparser
                        value = dateparser.parse(str(value)).isoformat()
                    except:
                        value = str(value)

                standard[std_field] = value

            elif std_field in self.defaults:
                # Use default value if field not found
                standard[std_field] = self.defaults[std_field]
            else:
                # Field missing and no default
                standard[std_field] = None

        # Add metadata
        standard['_bank']      = self.bank_name
        standard['_adapted']   = True
        standard['_timestamp'] = datetime.now().isoformat()

        return standard

    def validate(self, standard_event: dict) -> tuple:
        """
        Validate a translated event has all required fields.
        Returns (is_valid, list_of_missing_fields)
        """
        required = ['account_from', 'amount_paid', 'timestamp']
        missing  = [f for f in required if not standard_event.get(f)]
        return len(missing) == 0, missing

    def translate_batch(self, raw_events: list) -> list:
        """Translate a batch of events."""
        translated = []
        errors     = 0
        for event in raw_events:
            try:
                std = self.translate(event)
                valid, missing = self.validate(std)
                if valid:
                    translated.append(std)
                else:
                    errors += 1
            except Exception as e:
                errors += 1
        return translated, errors


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== Schema Adapter Test ===\n")

    # Test SBI adapter
    print("Testing SBI Finacle adapter...")
    sbi_adapter = SchemaAdapter('schema_configs/sbi_finacle.yaml')

    # Simulate a raw SBI transaction event
    sbi_event = {
        "DEBIT_ACCT_NO":   "SBI001234567",
        "CREDIT_ACCT_NO":  "SBI009876543",
        "TXN_AMOUNT":      "150000.00",
        "TXN_CCY":         "INR",
        "TXN_TYPE":        "NEFT",
        "TXN_DATE":        "2026-06-17 14:23:45",
        "UTR_NO":          "SBINR2026061700001",
        "BRANCH_CODE":     "SBI001",
        "DEST_BANK_CODE":  "HDFC002",
        "IS_FRAUD":        "0"
    }

    standard = sbi_adapter.translate(sbi_event)
    print("\nRaw SBI event:")
    for k, v in sbi_event.items():
        print(f"  {k}: {v}")
    print("\nStandard schema:")
    for k, v in standard.items():
        print(f"  {k}: {v}")

    valid, missing = sbi_adapter.validate(standard)
    print(f"\nValid: {valid}")

    print("\n" + "="*40)
    print("Testing Kotak CBS adapter...")
    kotak_adapter = SchemaAdapter('schema_configs/kotak_cbs.yaml')

    kotak_event = {
        "FROM_ACCOUNT_ID":  "KTK112233445",
        "TO_ACCOUNT_ID":    "KTK998877665",
        "TRANSACTION_AMT":  "75000.50",
        "CURRENCY_CODE":    "INR",
        "PAYMENT_MODE":     "IMPS",
        "CREATED_AT":       "2026-06-17T14:23:45.000Z",
        "TXN_REF_ID":       "KTKIMPS20260617001",
        "SOURCE_IFSC":      "KKBK0001234",
        "DEST_IFSC":        "HDFC0005678",
        "FRAUD_FLAG":       "0"
    }

    standard_k = kotak_adapter.translate(kotak_event)
    print("\nRaw Kotak event:")
    for k, v in kotak_event.items():
        print(f"  {k}: {v}")
    print("\nStandard schema:")
    for k, v in standard_k.items():
        print(f"  {k}: {v}")

    valid_k, missing_k = kotak_adapter.validate(standard_k)
    print(f"\nValid: {valid_k}")
    print("\nBoth banks translated to identical standard schema.")
    print("Zero code changes needed to switch between banks.")