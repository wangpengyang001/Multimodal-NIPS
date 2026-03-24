#!/usr/bin/env python3
import argparse
import logging
import sys
import os
from nids.system import NIDSSystem
import config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nids.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Network Intrusion Detection System')
    parser.add_argument('--interface', type=str, default=config.DATA_ACQUISITION['interface'], help='Network interface to monitor')
    parser.add_argument('--model', type=str, default=config.MODEL['model_path'], help='Path to trained model')
    parser.add_argument('--mode', type=str, choices=['detect', 'train'], default='detect', help='Operation mode')
    parser.add_argument('--dataset', type=str, default=None, help='Path to dataset for training')
    parser.add_argument('--epochs', type=int, default=config.TRAINING['epochs'], help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=config.TRAINING['batch_size'], help='Batch size for training')
    parser.add_argument('--lr', type=float, default=config.TRAINING['learning_rate'], help='Learning rate')
    
    args = parser.parse_args()
    
    try:
        nids = NIDSSystem(
            interface=args.interface,
            model_path=args.model
        )
        
        if args.mode == 'detect':
            logger.info('Starting NIDS in detection mode')
            nids.start_detection()
        elif args.mode == 'train':
            if not args.dataset:
                logger.error('Dataset path is required for training')
                sys.exit(1)
            logger.info('Starting NIDS in training mode')
            nids.train(args.dataset, args.epochs, args.batch_size, args.lr)
            
    except Exception as e:
        logger.error(f'Error starting NIDS: {e}')
        sys.exit(1)

if __name__ == '__main__':
    main()
