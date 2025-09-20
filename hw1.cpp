#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <unordered_set>
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

// Custom hash for State
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

// Global
queue<State> q;
mutex q_mutex;
condition_variable cv;
unordered_set<State, StateHash> visited;
mutex visited_mutex;
atomic<bool> solved(false);

State try_move(const State &s, int dy, int dx) {
    State result = s;
    result.valid = true;

    int y = s.pos.first, x = s.pos.second;
    int yy = y + dy, xx = x + dx;
    int yyy = yy + dy, xxx = xx + dx;

    if (s.board[yy][xx] == ' ') {
        result.board[yy][xx] = 'o';
    } else if (s.board[yy][xx] == '.') {
        result.board[yy][xx] = 'O';
    } else if (s.board[yy][xx] == '@') {
        result.board[yy][xx] = '!';
    } else if ((s.board[yy][xx] == 'x' || s.board[yy][xx] == 'X') &&
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
            if (line[x] == 'o' || line[x] == 'O' ||
                line[x] == '!' || line[x] == '@') {
                s.pos = {y, x};
            }
        }
        s.board.push_back(line);
        y++;
    }
    return s;
}

void worker() {
    vector<pair<int,int>> dirs = {{0,1},{0,-1},{1,0},{-1,0}};
    vector<char> moves = {'D','A','S','W'};

    while (!solved) {
        State cur;
        {
            unique_lock<mutex> lock(q_mutex);
            cv.wait(lock, [] { return !q.empty() || solved; });
            if (solved) return;
            cur = q.front();
            q.pop();
        }

        for (int i = 0; i < 4; i++) {
            State next = try_move(cur, dirs[i].first, dirs[i].second);
            if (!next.valid) continue;
            next.path = cur.path + moves[i];

            if (is_solved(next)) {
                cout << next.path << endl;
                solved = true;
                cv.notify_all();
                return;
            }

            {
                lock_guard<mutex> lock(visited_mutex);
                if (visited.find(next) != visited.end()) continue;
                visited.insert(next);
            }

            {
                lock_guard<mutex> lock(q_mutex);
                q.push(next);
            }
            cv.notify_one();
        }
    }
}

int main(int argc, char **argv) {
    string file(argv[1]);
    State start = loadstate(file);

    {
        lock_guard<mutex> lock(q_mutex);
        q.push(start);
        visited.insert(start);
    }

    int thread_count = 4;
    vector<thread> pool;
    for (int i = 0; i < thread_count; i++)
        pool.emplace_back(worker);

    cv.notify_all();
    for (auto &t : pool) t.join();

    if (!solved)
        cout << "No solution found." << endl;

    return 0;
}

