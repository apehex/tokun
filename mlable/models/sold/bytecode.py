import tensorflow as tf

# OPCODES #####################################################################

STOP = 0x00
EQ = 0x14
EXTCODECOPY = 0x3C
BLOCKHASH = 0x40
COINBASE = 0x41
PREVRANDAO = 0x44
JUMPDEST = 0x5B
PUSH1 = 0x60
PUSH32 = 0x7F
CREATE = 0xF0
CALLCODE = 0xF2
RETURN = 0xF3
DELEGATECALL = 0xF4
CREATE2 = 0xF5
REVERT = 0xFD
INVALID = 0xFE
SELFDESTRUCT = 0xFF

OPCODES = {
    0x00: 'STOP',
    0x01: 'ADD',
    0x02: 'MUL',
    0x03: 'SUB',
    0x04: 'DIV',
    0x05: 'SDIV',
    0x06: 'MOD',
    0x07: 'SMOD',
    0x08: 'ADDMOD',
    0x09: 'MULMOD',
    0x0a: 'EXP',
    0x0b: 'SIGNEXTEND',
    0x0c: 'INVALID',
    0x0f: 'Unused',
    0x10: 'LT',
    0x11: 'GT',
    0x12: 'SLT',
    0x13: 'SGT',
    0x14: 'EQ',
    0x15: 'ISZERO',
    0x16: 'AND',
    0x17: 'OR',
    0x18: 'XOR',
    0x19: 'NOT',
    0x1a: 'BYTE',
    0x1b: 'SHL',
    0x1c: 'SHR',
    0x1d: 'SAR',
    0x20: 'KECCAK256',
    0x21: 'INVALID',
    0x22: 'INVALID',
    0x23: 'INVALID',
    0x24: 'INVALID',
    0x25: 'INVALID',
    0x26: 'INVALID',
    0x27: 'INVALID',
    0x28: 'INVALID',
    0x29: 'INVALID',
    0x2a: 'INVALID',
    0x2b: 'INVALID',
    0x2c: 'INVALID',
    0x2d: 'INVALID',
    0x2e: 'INVALID',
    0x2f: 'INVALID',
    0x30: 'ADDRESS',
    0x31: 'BALANCE',
    0x32: 'ORIGIN',
    0x33: 'CALLER',
    0x34: 'CALLVALUE',
    0x35: 'CALLDATALOAD',
    0x36: 'CALLDATASIZE',
    0x37: 'CALLDATACOPY',
    0x38: 'CODESIZE',
    0x39: 'CODECOPY',
    0x3a: 'GASPRICE',
    0x3b: 'EXTCODESIZE',
    0x3c: 'EXTCODECOPY',
    0x3d: 'RETURNDATASIZE',
    0x3e: 'RETURNDATACOPY',
    0x3f: 'EXTCODEHASH',
    0x40: 'BLOCKHASH',
    0x41: 'COINBASE',
    0x42: 'TIMESTAMP',
    0x43: 'NUMBER',
    0x44: 'PREVRANDAO',
    0x45: 'GASLIMIT',
    0x46: 'CHAINID',
    0x47: 'SELFBALANCE',
    0x48: 'BASEFEE',
    0x49: 'BLOBHASH',
    0x4a: 'INVALID',
    0x4b: 'INVALID',
    0x4c: 'INVALID',
    0x4d: 'INVALID',
    0x4e: 'INVALID',
    0x4f: 'INVALID',
    0x48: 'BASEFEE',
    0x50: 'POP',
    0x51: 'MLOAD',
    0x52: 'MSTORE',
    0x53: 'MSTORE8',
    0x54: 'SLOAD',
    0x55: 'SSTORE',
    0x56: 'JUMP',
    0x57: 'JUMPI',
    0x58: 'GETPC',
    0x59: 'MSIZE',
    0x5a: 'GAS',
    0x5b: 'JUMPDEST',
    0x5c: 'TLOAD',
    0x5d: 'TSTORE',
    0x5e: 'MCOPY',
    0x5f: 'PUSH0',
    0x60: 'PUSH1',
    0x61: 'PUSH2',
    0x62: 'PUSH3',
    0x63: 'PUSH4',
    0x64: 'PUSH5',
    0x65: 'PUSH6',
    0x66: 'PUSH7',
    0x67: 'PUSH8',
    0x68: 'PUSH9',
    0x69: 'PUSH10',
    0x6a: 'PUSH11',
    0x6b: 'PUSH12',
    0x6c: 'PUSH13',
    0x6d: 'PUSH14',
    0x6e: 'PUSH15',
    0x6f: 'PUSH16',
    0x70: 'PUSH17',
    0x71: 'PUSH18',
    0x72: 'PUSH19',
    0x73: 'PUSH20',
    0x74: 'PUSH21',
    0x75: 'PUSH22',
    0x76: 'PUSH23',
    0x77: 'PUSH24',
    0x78: 'PUSH25',
    0x79: 'PUSH26',
    0x7a: 'PUSH27',
    0x7b: 'PUSH28',
    0x7c: 'PUSH29',
    0x7d: 'PUSH30',
    0x7e: 'PUSH31',
    0x7f: 'PUSH32',
    0x80: 'DUP1',
    0x81: 'DUP2',
    0x82: 'DUP3',
    0x83: 'DUP4',
    0x84: 'DUP5',
    0x85: 'DUP6',
    0x86: 'DUP7',
    0x87: 'DUP8',
    0x88: 'DUP9',
    0x89: 'DUP10',
    0x8a: 'DUP11',
    0x8b: 'DUP12',
    0x8c: 'DUP13',
    0x8d: 'DUP14',
    0x8e: 'DUP15',
    0x8f: 'DUP16',
    0x90: 'SWAP1',
    0x91: 'SWAP2',
    0x92: 'SWAP3',
    0x93: 'SWAP4',
    0x94: 'SWAP5',
    0x95: 'SWAP6',
    0x96: 'SWAP7',
    0x97: 'SWAP8',
    0x98: 'SWAP9',
    0x99: 'SWAP10',
    0x9a: 'SWAP11',
    0x9b: 'SWAP12',
    0x9c: 'SWAP13',
    0x9d: 'SWAP14',
    0x9e: 'SWAP15',
    0x9f: 'SWAP16',
    0xa0: 'LOG0',
    0xa1: 'LOG1',
    0xa2: 'LOG2',
    0xa3: 'LOG3',
    0xa4: 'LOG4',
    0xa5: 'INVALID',
    0xa6: 'INVALID',
    0xa7: 'INVALID',
    0xa8: 'INVALID',
    0xa9: 'INVALID',
    0xaa: 'INVALID',
    0xab: 'INVALID',
    0xac: 'INVALID',
    0xad: 'INVALID',
    0xae: 'INVALID',
    0xaf: 'INVALID',
    0xb0: 'JUMPTO',
    0xb1: 'JUMPIF',
    0xb2: 'JUMPSUB',
    0xb4: 'JUMPSUBV',
    0xb5: 'BEGINSUB',
    0xb6: 'BEGINDATA',
    0xb8: 'RETURNSUB',
    0xb9: 'PUTLOCAL',
    0xba: 'GETLOCAL',
    0xbb: 'INVALID',
    0xe0: 'INVALID',
    0xe1: 'SLOADBYTES',
    0xe2: 'SSTOREBYTES',
    0xe3: 'SSIZE',
    0xe4: 'INVALID',
    0xef: 'INVALID',
    0xf0: 'CREATE',
    0xf1: 'CALL',
    0xf2: 'CALLCODE',
    0xf3: 'RETURN',
    0xf4: 'DELEGATECALL',
    0xf5: 'CREATE2',
    0xf6: 'INVALID',
    0xf7: 'INVALID',
    0xf8: 'INVALID',
    0xf9: 'INVALID',
    0xfa: 'STATICCALL',
    0xfb: 'Unused',
    0xfc: 'TXEXECGAS',
    0xfd: 'REVERT',
    0xfe: 'INVALID',
    0xff: 'SELFDESTRUCT',}

HALTING = [STOP, RETURN, REVERT, INVALID, SELFDESTRUCT]

is_halting = lambda opcode: opcode in HALTING
is_push = lambda opcode: opcode >= PUSH1 and opcode <= PUSH32

# INSTRUCTIONS ################################################################

def instruction_length(opcode: int) -> int:
    return 1 + is_push(opcode) * (1 + opcode - PUSH1) # 1 byte for the opcode + n bytes of data

def disassemble(bytecode: bytes) -> str:
    __i = 0
    __a = ''
    while __i < len(bytecode):
        __len = instruction_length(opcode=bytecode[__i])
        __opcode = OPCODES.get(bytecode[__i], 'INVALID')
        __data = int(__len > 1) * (' ' + bytecode[__i + 1: __i + __len].hex())
        __a += __opcode + __data + '\n'
        __i = __i + __len
    return __a

# TOKENS ######################################################################

def one_hot(index: int, depth: int) -> list:
    __i = index % depth
    return __i * [0] + [1] + (depth - __i - 1) * [0]

# > ###########################################################################

def _tokenize_data(data: bytes) -> list:
    __bits = '{0:0>256b}'.format(int(data.hex(), 16) if data else 0) # expects at most a 32-byte word of data
    return [int(__b) for __b in __bits[::-1]] # little endian

def _tokenize_instruction(chunk: bytes) -> list:
    return one_hot(index=chunk[0], depth=256) + _tokenize_data(data=chunk[1:])

def tokenize(bytecode: bytes) -> iter:
    __i = 0
    while __i < len(bytecode):
        __len = instruction_length(opcode=bytecode[__i])
        yield _tokenize_instruction(chunk=bytecode[__i:__i+__len])
        __i = __i + __len

# < ###########################################################################

def interpret(output: tf.Tensor) -> tf.Tensor:
    __dim = output.shape[-1] // 2
    __output = tf.reshape(tensor=output, shape=(-1, 2, __dim))
    __opcode = tf.cast(x=tf.one_hot(indices=tf.argmax(input=__output[:, 0, :], axis=-1, output_type=tf.dtypes.int32), depth=__dim), dtype=tf.dtypes.float32)
    __data = tf.cast(x=__output[:, 1, :] >= 0.5, dtype=tf.dtypes.float32)
    return tf.concat(values=[__opcode, __data], axis=-1)

def _detokenize_instruction(x: list) -> bytes:
    # dimension of the opcode / data encodings (256)
    __dim = len(x) // 2
    # returns the first one, for the opcode
    __opcode = x.index(1)
    # length of the HEX encoded data
    __len = 2 * max(0, instruction_length(opcode=__opcode) - 1)
    # format for the data
    __hex = (__len > 0) * '{{0:0>{length}x}}'.format(length=__len)
    # data
    __data = int(''.join(str(__b) for __b in x[__dim:])[::-1], 2)
    return bytes([__opcode]) + bytes.fromhex(__hex.format(__data))

def detokenize(x: list, merge: bool=True) -> bytes:
    __instructions = [_detokenize_instruction(x=__i) for __i in x]
    return b''.join(__instructions) if merge else __instructions

# print("\033[48;5;200mHello\033[0m")
