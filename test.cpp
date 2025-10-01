#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <queue>
#include <pthread.h>
#include <unordered_set>
#include <atomic>
using namespace std;
struct State {
    vector<string> board;
    pair<int,int> pos; 
    bool valid;
    string path;

    bool operator==(const State &other) const {
        return pos == other.pos && board == other.board;
    }
};

struct StateHash {
    size_t operator()(const State &s) const {
        size_t seed = 0;
        for (auto &row : s.board) {
            for (char c : row) {
                seed ^= hash<char>()(c) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
        }
        seed ^= hash<int>()(s.pos.first);
        seed ^= hash<int>()(s.pos.second);
        return seed;
    }
};

// Globals
queue<State> q;
pthread_mutex_t q_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cv = PTHREAD_COND_INITIALIZER;
unordered_set<State, StateHash> visited;
pthread_mutex_t visited_mutex = PTHREAD_MUTEX_INITIALIZER;
atomic<bool> solved(false);
void printState(State s){
	pthread_mutex_lock(&q_mutex);
	for (auto &row : s.board) {
            for (char c : row) {
		cout << c;
            }
	cout << endl;
	}
	pthread_mutex_unlock(&q_mutex);
}

// ---------- Game Logic ----------
State try_move(const State &s, int dy, int dx) {
    State result = s;
    result.valid = true;

    int y = s.pos.first, x = s.pos.second;
    int yy = y + dy, xx = x + dx;
    int yyy = yy + dy, xxx = xx + dx;

    int rows = s.board.size();
    int cols = s.board[0].size();

    // bounds check
    if (yy < 0 || yy >= rows || xx < 0 || xx >= cols) {
        result.valid = false;
        return result;
    }
    if ((s.board[yy][xx] == 'x' || s.board[yy][xx] == 'X') &&
        (yyy < 0 || yyy >= rows || xxx < 0 || xxx >= cols)) {
        result.valid = false;
        return result;
    }

    if (s.board[yy][xx] == ' ') result.board[yy][xx] = 'o';
    else if (s.board[yy][xx] == '.') result.board[yy][xx] = 'O';
    else if (s.board[yy][xx] == '@') result.board[yy][xx] = '!';
    else if ((s.board[yy][xx] == 'x' || s.board[yy][xx] == 'X') &&
             (s.board[yyy][xxx] == ' ' || s.board[yyy][xxx] == '.')) {
        result.board[yy][xx] = (s.board[yy][xx] == 'x') ? 'o' : 'O';
        result.board[yyy][xxx] = (s.board[yyy][xxx] == ' ') ? 'x' : 'X';
    } else {
        result.valid = false;
        return result;
    }

    if (s.board[y][x] == 'o') result.board[y][x] = ' ';
    else if (s.board[y][x] == '!') result.board[y][x] = '@';
    else result.board[y][x] = '.';

    result.pos = {yy, xx};
    return result;
}

bool is_solved(const State &s) {
    for (auto &row : s.board)
        for (char c : row)
            if (c == '.' || c == 'O') return false;
    return true;
}

// ---------- State Loading ----------
State loadstate(const string &filename) {
    ifstream file(filename);
    if (!file) {
        cerr << "Error: cannot open file " << filename << "\n";
        exit(1);
    }

    State s;
    s.board.clear();
    s.valid = true;
    string line;
    int y = 0;
    while (getline(file, line)) {
        for (int x = 0; x < (int)line.size(); x++) {
            if (line[x] == 'o' || line[x] == 'O' || line[x] == '!' || line[x] == '@') {
                s.pos = {y, x};
            }
        }
        s.board.push_back(line);
        y++;
    }
    return s;
}

// Helper lambdas use the current board; goals are ('.' or 'O' or 'X'), walls are '#'
bool is_deadlock(const State& s) {
    const auto& b = s.board;
    const int R = (int)b.size();
    const int C = (int)b[0].size();
    auto inb = [&](int y, int x){ return y>=0 && y<R && x>=0 && x<C; };
    auto isWall = [&](int y, int x){ return inb(y,x) && b[y][x] == '#'; };
    auto isGoal = [&](int y, int x){
        if (!inb(y,x)) return false;
        char c = b[y][x];
        // Treat 'X' (box on goal) as a goal cell for taboo computation
        return (c == '.' || c == 'O' || c == 'X');
    };
    auto isFloorLike = [&](int y, int x){
        return inb(y,x) && b[y][x] != '#';
    };

    // 1) Taboo grid (false by default)
    std::vector<std::vector<bool>> taboo(R, std::vector<bool>(C, false));

    // 1a) Corner taboo (non-goal floor-like with two perpendicular walls)
    for (int y = 1; y < R-1; ++y) {
        for (int x = 1; x < C-1; ++x) {
            if (!isFloorLike(y,x) || isGoal(y,x)) continue;
            bool up = isWall(y-1,x), dn = isWall(y+1,x);
            bool lf = isWall(y,x-1), rt = isWall(y,x+1);
            if ((up && lf) || (up && rt) || (dn && lf) || (dn && rt)) {
                taboo[y][x] = true;
            }
        }
    }

    // Internal helpers to extend taboo along walls between corners that share the same wall side
    auto extend_horizontal = [&](int y, bool use_top_wall){
        // scan row y for pairs of "corner cells that also touch the chosen wall side"
        int x = 1;
        while (x < C-1) {
            // find first corner with required side-wall contact
            while (x < C-1) {
                if (taboo[y][x] && !isGoal(y,x) &&
                    (use_top_wall ? isWall(y-1,x) : isWall(y+1,x))) break;
                ++x;
            }
            if (x >= C-1) break;
            int x1 = x; ++x;

            // find the next corner with same side-wall contact
            while (x < C-1) {
                if (taboo[y][x] && !isGoal(y,x) &&
                    (use_top_wall ? isWall(y-1,x) : isWall(y+1,x))) break;
                ++x;
            }
            if (x >= C-1) break;
            int x2 = x;

            // check the segment (x1, x2) (excluding corners) sticks to that wall side,
            // has no internal walls, and contains no goals
            bool ok = true, has_goal = false;
            for (int k = x1 + 1; k <= x2 - 1; ++k) {
                if (!isFloorLike(y,k)) { ok = false; break; }                  // hits a wall inside
                if (use_top_wall ? !isWall(y-1,k) : !isWall(y+1,k)) { ok=false; break; } // not hugging the same wall
                if (isGoal(y,k)) has_goal = true;
            }
            if (ok && !has_goal) {
                for (int k = x1 + 1; k <= x2 - 1; ++k)
                    if (!isGoal(y,k)) taboo[y][k] = true;
            }
            // continue search after x2
            ++x;
        }
    };

    auto extend_vertical = [&](int x, bool use_left_wall){
        int y = 1;
        while (y < R-1) {
            while (y < R-1) {
                if (taboo[y][x] && !isGoal(y,x) &&
                    (use_left_wall ? isWall(y,x-1) : isWall(y,x+1))) break;
                ++y;
            }
            if (y >= R-1) break;
            int y1 = y; ++y;

            while (y < R-1) {
                if (taboo[y][x] && !isGoal(y,x) &&
                    (use_left_wall ? isWall(y,x-1) : isWall(y,x+1))) break;
                ++y;
            }
            if (y >= R-1) break;
            int y2 = y;

            bool ok = true, has_goal = false;
            for (int k = y1 + 1; k <= y2 - 1; ++k) {
                if (!isFloorLike(k,x)) { ok = false; break; }
                if (use_left_wall ? !isWall(k,x-1) : !isWall(k,x+1)) { ok=false; break; }
                if (isGoal(k,x)) has_goal = true;
            }
            if (ok && !has_goal) {
                for (int k = y1 + 1; k <= y2 - 1; ++k)
                    if (!isGoal(k,x)) taboo[k][x] = true;
            }
            ++y;
        }
    };

    // 1b) Extend taboo runs along each wall side
    for (int y = 1; y < R-1; ++y) {
        extend_horizontal(y, /*use_top_wall=*/true);
        extend_horizontal(y, /*use_top_wall=*/false);
    }
    for (int x = 1; x < C-1; ++x) {
        extend_vertical(x, /*use_left_wall=*/true);
        extend_vertical(x, /*use_left_wall=*/false);
    }

    // 2) If any box 'x' is on a taboo cell, the state is dead
    for (int y = 1; y < R-1; ++y) {
        for (int x = 1; x < C-1; ++x) {
            if (b[y][x] == 'x' && taboo[y][x]) {
                return true;
            }
        }
    }
    return false;
}

// ---------- Worker Thread ----------
void* worker(void*) {
    vector<pair<int,int>> dirs = {{0,1},{0,-1},{1,0},{-1,0}};
    vector<char> moves = {'D','A','S','W'};

    while (!solved) {
        State cur;
        {
            pthread_mutex_lock(&q_mutex);
            while (q.empty() && !solved) {
                pthread_cond_wait(&cv, &q_mutex);
            }
            if (solved) {
                pthread_mutex_unlock(&q_mutex);
                return nullptr;
            }
            cur = q.front();
            q.pop();
            pthread_mutex_unlock(&q_mutex);
        }
	printState(cur);
        for (int i = 0; i < 4; i++) {
            if (solved) return nullptr;

            State next = try_move(cur, dirs[i].first, dirs[i].second);
            if (!next.valid) continue;
            next.path = cur.path + moves[i];

            // Deadlock check
            if (is_deadlock(next)) continue;

            if (is_solved(next)) {
                if (!solved.exchange(true)) {
                    cout << "Solved! Path: " << next.path << endl;
                    pthread_cond_broadcast(&cv);
                }
                return nullptr;
            }

            bool seen = false;
            pthread_mutex_lock(&visited_mutex);
            if (visited.find(next) != visited.end()) seen = true;
            else visited.insert(next);
            pthread_mutex_unlock(&visited_mutex);

            if (!seen) {
                pthread_mutex_lock(&q_mutex);
                q.push(next);
                pthread_mutex_unlock(&q_mutex);
                pthread_cond_signal(&cv);
            }
        }
    }
    return nullptr;
}

// ---------- Main ----------
int main(int argc, char **argv) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " map.txt\n";
        return 1;
    }

    string file(argv[1]);
    State start = loadstate(file);

    cout << is_deadlock(start) << endl;
/*
    pthread_mutex_lock(&q_mutex);
    q.push(start);
    visited.insert(start);
    pthread_mutex_unlock(&q_mutex);

    int thread_count = 8; // adjust based on machine
    vector<pthread_t> threads(thread_count);
    for (int i = 0; i < thread_count; i++) {
        pthread_create(&threads[i], nullptr, worker, nullptr);
    }

    pthread_cond_broadcast(&cv);

    for (int i = 0; i < thread_count; i++) {
        pthread_join(threads[i], nullptr);
    }

    if (!solved)
        cout << "No solution found." << endl;
*/
    return 0;
}

