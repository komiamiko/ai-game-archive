#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <vector>
#include <string>
#include <cmath>

#define X_STEP 10
#define ROOT_HALF 0.25
#define PITCH_WIDTH 11
#define PITCH_HEIGHT 17
#define MAX_ITR 5
typedef std::pair<std::pair<std::pair<float,float>,std::pair<float,float> >,int> disc_t;

std::vector<std::pair<float,float> > gen_valid_moves(std::vector<std::pair<float, float> >&discs, const float&x, const float&y) {
  int x_seed = rand() % X_STEP;
  int y_seed = rand() % X_STEP;
  std::vector<std::pair<float,float> > rtn;
  for (int _mx = x_seed; _mx < x_seed + X_STEP; _mx++) {
    for (int _my = y_seed; _my < y_seed + X_STEP; _my++) {
      int mx = (_mx%X_STEP)-X_STEP/2,
          my = (_my%X_STEP)-X_STEP/2;
      //printf("%d, %d : %d, %d\n",_mx, _my, mx, my);
      float norm = ROOT_HALF / (std::sqrt(mx*mx + my*my)+1e-30);
      float xp = x + mx*norm, yp = y + my*norm;
      bool ok = true;
      //printf("%f, %f\n", xp, yp);
      for (auto& disc : discs) {
        float xd = abs(disc.first - xp);
        float yd = abs(disc.second - yp);
        
        if (xd*xd + yd*yd < 1.1 || xp <= 1 || yp <= 1 || xp >= PITCH_WIDTH - 1 || yp >= PITCH_HEIGHT - 1) {
          ok = false; break;
        }
      }
      if (ok) rtn.push_back(std::make_pair(mx*norm,my*norm));
    }
  }
  return rtn;
}

std::vector<std::pair<float, float> > filter_valid_moves(std::vector<std::pair<float, float> >&discs, std::vector<std::pair<float, float> >&moves, const float&x, const float&y) {
  for (auto it = moves.begin(); it < moves.end();) {
    float x1 = it -> first + x,
          y1 = it -> second + y;
    if (gen_valid_moves(discs, x1, y1).size() == 0) {
      moves.erase(it);
    } else it++;
  }
  return moves;
}

std::vector<std::pair<float,float> > move_discs(std::vector<disc_t>& discs) {
  std::vector<std::pair<float,float> > rtn;
  for (auto& disc : discs) {
    disc.first.first.first += disc.first.second.first;
    disc.first.first.second += disc.first.second.second;
    if (disc.first.first.first > PITCH_WIDTH) {
      disc.second--;
      disc.first.first.first = (2*PITCH_WIDTH)-disc.first.first.first;
    } else if (disc.first.first.first < 0) {
      disc.second--;
      disc.first.first.first = -disc.first.first.first;
    }
    if (disc.first.first.second > PITCH_HEIGHT) {
      disc.second--;
      disc.first.first.second = (2*PITCH_HEIGHT)-disc.first.first.second;
    } else if (disc.first.first.second < 0) {
      disc.second--;
      disc.first.first.second = -disc.first.first.second;
    }
    if (disc.second >= 0) rtn.push_back(disc.first.first);
  }
  return rtn;
}


static PyObject* tj_get_action(PyObject* self, PyObject* args) {
  PyObject* disc_list;
  int disc_count, op_ch, me_ch;
  float op_x, op_y, me_x, me_y;
  if (!PyArg_ParseTuple(args, "Oiffiffi", &disc_list, &disc_count, &op_x, &op_y, &op_ch, &me_x, &me_y, &me_ch)) return NULL;
  std::vector<disc_t> discs;
  for (int i = 0; i < disc_count; i++) {
    float opx, opy, ovx, ovy;
    int odc;
    PyObject* disc_obj = PyList_GetItem(disc_list, i);
    PyArg_ParseTuple(disc_obj, "ffffi", &opx, &opy, &ovx, &ovy, &odc);
    disc_t disc = std::make_pair(
      std::make_pair(
        std::make_pair(opx,opy),
        std::make_pair(ovx,ovy)),
      odc);
    discs.push_back(disc);
  }
  float movex,movey,throwx,throwy;
  std::vector<std::pair<float, float> > dp = move_discs(discs);
  std::vector<std::pair<float, float> > pos = gen_valid_moves(dp, me_x, me_y);
  auto pos_last = pos;
  int itr = 0;
  while (pos.size() > 0 && disc_count > 0 && itr++ < MAX_ITR) {
#ifndef _RELEASE
    printf("size: %d\n", pos.size());
    printf("%d\n", itr);
#endif
    pos_last = pos;
    dp = move_discs(discs);
    pos = filter_valid_moves(dp, pos, me_x, me_y);
  }
  if (pos.size() != 0) {
    int r = random()%pos.size();
    movex = pos[r].first;
    movey = pos[r].second;
  } else if (pos_last.size() != 0) {
    int r = random()%pos_last.size();
    movex = pos_last[r].first;
    movey = pos_last[r].second;
  } else {
    movex = 0;
    movey = 0;
  }
  throwx = op_x - me_x;
  throwy = op_y - me_y;
  
  PyObject* rtn = PyTuple_Pack(4,
    PyFloat_FromDouble(movex),
    PyFloat_FromDouble(movey),
    PyFloat_FromDouble(throwx),
    PyFloat_FromDouble(throwy));
  return rtn;
}

static PyObject* TjError;

void map_disc_positions(int hmm) {
  // Will be used to track the history of disc positions
}

static PyObject* tj_reset(PyObject *self, PyObject *args)
{
    int is_new_opponent;
    if (!PyArg_ParseTuple(args, "b", &is_new_opponent)) return NULL;
    if (is_new_opponent) { } // Maybe make some new info about opponent history?
    return PyBool_FromLong(1);
}

static PyMethodDef TjMethods[] = {
    {"reset",  tj_reset, METH_VARARGS,
     "Reset tj's bot"},
     {"get_action",  tj_get_action, METH_VARARGS,
     "Get tj's bot's action"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef tjmodule = {
    PyModuleDef_HEAD_INIT,
    "tj",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    TjMethods
};



PyMODINIT_FUNC
PyInit_tj(void)
{
    PyObject* m;

    m = PyModule_Create(&tjmodule);
    if (m == NULL)
        return NULL;

    TjError = PyErr_NewException("tj.error", NULL, NULL);
    Py_XINCREF(TjError);
    if (PyModule_AddObject(m, "error", TjError) < 0) {
        Py_XDECREF(TjError);
        Py_CLEAR(TjError);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}


