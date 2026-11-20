"""Second move validator for checking the official five-ply scores."""

import json
import re

import chess

UCI = re.compile(r"[a-h][1-8][a-h][1-8]")


def attacks(board, origin, target, piece, swap, capture):
    x, y = origin
    dx, dy = target[0] - x, target[1] - y
    ax, ay = abs(dx), abs(dy)
    if ax == ay == 0:
        return False
    kind = piece.lower()
    if swap:
        kind = {'b': 'n', 'n': 'b'}.get(kind, kind)
    if kind == 'k':
        return max(ax, ay) == 1
    if kind == 'n':
        return sorted((ax, ay)) == [1, 2]
    if kind == 'p':
        advance = 1 if piece.isupper() else -1
        if capture:
            return ax == 1 and dy == advance
        return dx == 0 and (dy == advance or (dy == 2 * advance
            and y == (1 if advance == 1 else 6) and (x, y + advance) not in board))
    orthogonal = dx == 0 or dy == 0
    diagonal = ax == ay
    if not ((kind in 'rq' and orthogonal) or (kind in 'bq' and diagonal)):
        return False
    step_x, step_y = (dx > 0) - (dx < 0), (dy > 0) - (dy < 0)
    return all((x + j * step_x, y + j * step_y) not in board for j in range(1, max(ax, ay)))


def safe_after_move(board, origin, dest, white, rule):
    piece, target = board.get(origin), board.get(dest)
    if piece is None:
        return 'empty origin', None
    if piece.isupper() != white:
        return 'wrong side', None
    if target and target.isupper() == white:
        return 'own piece at destination', None
    if target and target.lower() == 'k':
        return 'king capture', None
    if piece.lower() == 'p' and dest[1] in (0, 7):
        return 'excluded promotion', None
    swap = rule == 'bishop_knight_swap'
    if not attacks(board, origin, dest, piece, swap, target is not None):
        return 'movement geometry or blocked path', None
    new = board.copy()
    del new[origin]
    new[dest] = piece
    king = next(square for square, p in new.items() if p == ('K' if white else 'k'))
    if any(attacks(new, s, king, p, swap, True) for s, p in new.items() if p.isupper() != white):
        return 'king in check after move', None
    return None, new


def normalize_notation(raw_move, board, white, rule):
    text = raw_move.strip().rstrip('+#!?')
    coordinate = re.fullmatch(r'[KQRBNPkqrbnp]?([a-h][1-8])[-x:> ]*([a-h][1-8])', text)
    if coordinate:
        return ''.join(coordinate.groups())
    san = re.fullmatch(r'([KQRBN])?([a-h])?([1-8])?(x)?([a-h][1-8])', text)
    if not san:
        return None
    kind, file, rank, capture, target = san.groups()
    dest = (ord(target[0]) - 97, int(target[1]) - 1)
    if (dest in board) != bool(capture):
        return None
    candidates = [source for source, piece in board.items()
                  if piece.isupper() == white and piece.upper() == (kind or 'P')
                  and (file is None or ord(file) - 97 == source[0])
                  and (rank is None or int(rank) - 1 == source[1])
                  and safe_after_move(board, source, dest, white, rule)[0] is None]
    if len(candidates) != 1:
        return None
    x, y = candidates[0]
    return f'{chr(97 + x)}{1 + y}{target}'


def independent_replay(fen, rule, moves, *, relaxed=False):
    initial = chess.Board(fen)
    board = {(s % 8, s // 8): p.symbol() for s, p in initial.piece_map().items()}
    white = initial.turn
    for index, move in enumerate(moves, 1):
        if relaxed and isinstance(move, str) and not UCI.fullmatch(move):
            move = normalize_notation(move, board, white, rule)
        if not isinstance(move, str) or not UCI.fullmatch(move):
            return index, 'malformed coordinate'
        origin = ord(move[0]) - 97, int(move[1]) - 1
        dest = ord(move[2]) - 97, int(move[3]) - 1
        issue, new = safe_after_move(board, origin, dest, white, rule)
        if issue:
            return index, issue
        if rule == 'mandatory_capture' and dest not in board:
            if any(safe_after_move(board, source, target, white, rule)[0] is None
                   for source in board if board[source].isupper() == white
                   for target in board if board[target].isupper() != white):
                return index, 'mandatory capture skipped'
        board, white = new, not white
    return None, None


def tolerant_moves(raw, moves):
    decoded = list(moves) if isinstance(moves, list) else []
    if not decoded and raw:
        for match in re.finditer(r'\{', raw):
            try:
                candidate, _ = json.JSONDecoder().raw_decode(raw[match.start():])
            except ValueError:
                continue
            if isinstance(candidate, dict) and isinstance(candidate.get('moves'), list):
                decoded = candidate['moves']
                break
    if len(decoded) == 5 and all(isinstance(m, str) for m in decoded):
        normalized = [re.sub(r'[\s\-x:>]+', '', m.strip().lower()) for m in decoded]
        if all(UCI.fullmatch(m) for m in normalized):
            return normalized
    if raw:
        return re.findall(r'(?<![a-z0-9])([a-h][1-8][a-h][1-8])(?![a-z0-9])', raw.lower())
    return []
