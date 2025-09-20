#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

// =================== STATE DEFINITION ===================
struct State {
    vector<string> board;
    pair<int, int> pos;   // player position (y, x)
    string path;          // moves leading here
    bool valid;

    bool operator==(const State &other) const {
        return pos == other.pos && board == other.board;
    }
};

// =================== HASH FUNCTION ===================
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

// =================== GAME LOGIC ===================
State try_move(const State &s, int dy, int dx) {
	//cout << dy << " " << dx << endl;
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
            if (line[x] == 'o' || line[x] == 'O' || line[x] == '!') {
                s.pos = {y, x};
            }
        }
        s.board.push_back(line);
        y++;
    }
    return s;
}

// =================== PARALLEL BFS SOLVER ===================
void solve_parallel(const State &start) {
    unordered_set<State, StateHash> visited;
    queue<State> frontier;

    visited.insert(start);
    frontier.push(start);

    atomic<bool> solved(false);

    vector<pair<int,int>> dirs = {{0,1},{0,-1},{1,0},{-1,0}};
    vector<char> moves = {'D','A','S','W'};

    while (!frontier.empty() && !solved) {
        int frontier_size = frontier.size();
        vector<State> current_level;

        // Extract all nodes in current BFS level
        for (int i = 0; i < frontier_size; i++) {
            current_level.push_back(frontier.front());
            frontier.pop();
        }

        vector<State> next_level;

        // Parallelize expansion
        #pragma omp parallel
        {
            vector<State> local_next;

            #pragma omp for nowait
            for (int i = 0; i < (int)current_level.size(); i++) {
                State cur = current_level[i];

                for (int d = 0; d < 4; d++) {
                    State next = try_move(cur, dirs[d].first, dirs[d].second);
                    if (!next.valid) continue;

                    next.path = cur.path + moves[d];

                    if (is_solved(next)) {
                        #pragma omp critical
                        {
                            if (!solved) {
                                cout << next.path << endl;
                                solved = true;
                            }
                        }
                        continue;
                    }

                    // visited check (thread-safe)
                    bool already_seen = false;
                    #pragma omp critical
                    {
                        if (visited.find(next) != visited.end()) {
                            already_seen = true;
                        } else {
                            visited.insert(next);
                        }
                    }

                    if (!already_seen) {
                        local_next.push_back(next);
                    }
                }
            }

            // Merge local_next into global next_level
            #pragma omp critical
            {
                next_level.insert(next_level.end(),
                                  local_next.begin(), local_next.end());
            }
        } // end parallel

        // push next frontier
        for (auto &s : next_level) {
            frontier.push(s);
        }
    }

    if (!solved) {
        cout << "No solution found." << endl;
    }
}

// =================== MAIN ===================
int main(int argc, char **argv) {
    State s = loadstate(argv[1]);

    double start_time = omp_get_wtime();
    solve_parallel(s);
    double end_time = omp_get_wtime();

    //cout << "Elapsed: " << (end_time - start_time) << " sec\n";
    return 0;
}

