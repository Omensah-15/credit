from web3 import Web3
import json
import os
from dotenv import load_dotenv
from typing import Dict, Any

# Load environment variables
load_dotenv()

class BlockchainManager:
    def __init__(self):
        # Initialize Web3 connection
        self.provider_url = os.getenv('WEB3_PROVIDER_URL', 'http://127.0.0.1:8545')
        self.w3 = Web3(Web3.HTTPProvider(self.provider_url))
        
        # Load contract ABI and address
        self.contract_address = os.getenv('CONTRACT_ADDRESS', '')
        
        # Load contract ABI
        try:
            with open('contracts/VerificationContract.json', 'r') as f:
                contract_data = json.load(f)
                self.contract_abi = contract_data['abi']
        except FileNotFoundError:
            # Fallback to simple ABI
            self.contract_abi = [
                {
                    "inputs": [
                        {"internalType": "string", "name": "applicantId", "type": "string"},
                        {"internalType": "string", "name": "dataHash", "type": "string"},
                        {"internalType": "uint256", "name": "riskScore", "type": "uint256"},
                        {"internalType": "string", "name": "riskCategory", "type": "string"},
                        {"internalType": "uint256", "name": "probabilityOfDefault", "type": "uint256"}
                    ],
                    "name": "storeVerificationResult",
                    "outputs": [],
                    "stateMutability": "nonpayable",
                    "type": "function"
                },
                {
                    "inputs": [{"internalType": "string", "name": "applicantId", "type": "string"}],
                    "name": "getVerificationResult",
                    "outputs": [
                        {"internalType": "string", "name": "dataHash", "type": "string"},
                        {"internalType": "uint256", "name": "riskScore", "type": "uint256"},
                        {"internalType": "string", "name": "riskCategory", "type": "string"},
                        {"internalType": "uint256", "name": "probabilityOfDefault", "type": "uint256"},
                        {"internalType": "uint256", "name": "timestamp", "type": "uint256"}
                    ],
                    "stateMutability": "view",
                    "type": "function"
                }
            ]
        
        # Set up account
        self.account_address = os.getenv('ACCOUNT_ADDRESS', '')
        self.private_key = os.getenv('PRIVATE_KEY', '')
        
        # Initialize contract
        if self.contract_address:
            self.contract = self.w3.eth.contract(
                address=self.contract_address,
                abi=self.contract_abi
            )
        else:
            self.contract = None
    
    def store_verification_result(self, applicant_id: str, data_hash: str, 
                                 risk_score: int, risk_category: str, 
                                 probability_of_default: float) -> str:
        """
        Store verification result on the blockchain.
        """
        if not self.contract:
            return "Error: Contract not initialized"
            
        try:
            # Convert probability to integer for blockchain storage
            probability_int = int(probability_of_default * 10000)
            
            # Build transaction
            transaction = self.contract.functions.storeVerificationResult(
                applicant_id, data_hash, risk_score, risk_category, probability_int
            ).build_transaction({
                'from': self.account_address,
                'nonce': self.w3.eth.get_transaction_count(self.account_address),
                'gas': 2000000,
                'gasPrice': self.w3.to_wei('50', 'gwei')
            })
            
            # Sign transaction
            signed_txn = self.w3.eth.account.sign_transaction(transaction, private_key=self.private_key)
            
            # Send transaction
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            # Wait for transaction receipt
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            return receipt.transactionHash.hex()
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def get_verification_result(self, applicant_id: str) -> Dict[str, Any]:
        """
        Retrieve verification result from the blockchain.
        """
        if not self.contract:
            return {'error': 'Contract not initialized'}
            
        try:
            result = self.contract.functions.getVerificationResult(applicant_id).call()
            
            return {
                'data_hash': result[0],
                'risk_score': result[1],
                'risk_category': result[2],
                'probability_of_default': result[3] / 10000.0,
                'timestamp': result[4]
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def is_connected(self) -> bool:
        """Check if connected to blockchain."""
        return self.w3.is_connected()
