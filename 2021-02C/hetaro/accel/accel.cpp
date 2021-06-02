#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <cmath>
#include <vector>

struct disc_t {
  float data[5];
};

struct model_buffer {
  float reward; // 0
  float done; // 1
  float rem_ticks; // 2
  float observation[2][11][7]; // 3
  float action[2][8]; // 157
  float hp[2]; // 173
  float charge[2]; // 175
  float hero_pos[2][2]; // 177
  float discs[2][50][5]; // 181
  // 681
};
// total 681

static PyObject* accel_new_match(PyObject* self, PyObject* args) {
  if (!PyArg_ParseTuple(args, ""))
        return NULL;

  PyObject* nfbuffer_py = PyList_New(0);

  model_buffer model;
  float* model_f = (float*)(&model);

  model.reward = 0;
  model.done = 0;
  model.rem_ticks = 1000;
  for(int i = 0; i < 2; ++i) {
    model.observation[i][0][0] = 1;
    model.observation[i][0][1] = model.hero_pos[i][0] = i==0?1:16;
    model.observation[i][0][2] = model.hero_pos[i][1] = i==0?1:10;
    model.observation[i][0][3] = 0;
    model.observation[i][0][4] = 0;
    model.observation[i][0][5] = model.hp[i] = 5;
    model.observation[i][0][6] = model.charge[i] = 1;
    for(int j = 1; j < 11; ++j) {
      for(int k = 0; k < 7; ++k) {
        model.observation[i][j][k] = 0;
      }
    }
    for(int j = 0; j < 8; ++j) {
      model.action[i][j] = 0;
    }
    for(int j = 0; j < 50; ++j) {
      for(int k = 0; k < 5; ++k) {
        model.discs[i][j][k] = -999999;
      }
    }
  }

  for(int i = 0; i < 681; ++i) {
    PyObject* fbi = PyFloat_FromDouble((double)(model_f[i]));
    PyList_Append(nfbuffer_py, fbi);
  }

  return nfbuffer_py;
}

static PyObject* accel_update(PyObject* self, PyObject* args) {
  PyObject* fbuffer_py;

  if (!PyArg_ParseTuple(args, "O", &fbuffer_py))
        return NULL;

  PyObject* nfbuffer_py = PyList_New(0);

  model_buffer model;
  float* model_f = (float*)(&model);

  for(int i = 0; i < 681; ++i) {
    PyObject* fbi = PyList_GetItem(fbuffer_py, i);
    double argif = PyFloat_AsDouble(fbi);
    model_f[i] = (float)(argif);
  }

  // copy in

  std::vector<disc_t> discs[2];

  for(int i = 0; i < 2; ++i) {
    for(int j = 0; j < 50; ++j) {
      if(model.discs[i][j][0] < -100)break;
      disc_t ijdisc = {model.discs[i][j][0], model.discs[i][j][1], model.discs[i][j][2], model.discs[i][j][3], model.discs[i][j][4]};
      discs[i].push_back(ijdisc);
    }
  }

  // throw discs

  for(int i = 0; i < 2; ++i) {
    for(int j = 0; j < 2; ++j) {
      if(model.action[i][3*j+2] > 0 && model.charge[i] >= 10) {
        model.charge[i] -= 10;
        float dx = model.action[i][3*j+3];
        float dy = model.action[i][3*j+4];
        float mul = 0.5f / hypotf(dx, dy);
        dx *= mul;
        dy *= mul;
        disc_t new_disc = {model.hero_pos[i][0], model.hero_pos[i][1], dx, dy, 4};
        discs[1-i].push_back(new_disc);
      }
    }
  }

  // do movement

  for(int i = 0; i < 2; ++i) {
    {
      float dx = model.action[i][0];
      float dy = model.action[i][1];
      float mag = hypotf(dx, dy);
      float mul = 0.25f;
      if(mag > 1) {
        mul /= mag;
      }
      float px = model.hero_pos[i][0];
      float py = model.hero_pos[i][1];
      px += dx * mul;
      py += dy * mul;
      if(px < 1) px = 1;
      if(px > 16) px = 16;
      if(py < 1) py = 1;
      if(py > 10) py = 10;
      model.hero_pos[i][0] = px;
      model.hero_pos[i][1] = py;
    }
    std::vector<disc_t>& idiscs = discs[i];
    for(int j = idiscs.size() - 1; j >= 0; --j) {
      disc_t& jdisc = idiscs[j];
      jdisc.data[0] += jdisc.data[2];
      jdisc.data[1] += jdisc.data[3];
      if(jdisc.data[0] < 0) {
        jdisc.data[0] = -jdisc.data[0];
        jdisc.data[2] = -jdisc.data[2];
        jdisc.data[4] -= 1;
      }
      if(jdisc.data[0] > 17) {
        jdisc.data[0] = 17*2-jdisc.data[0];
        jdisc.data[2] = -jdisc.data[2];
        jdisc.data[4] -= 1;
      }
      if(jdisc.data[1] < 0) {
        jdisc.data[1] = -jdisc.data[1];
        jdisc.data[3] = -jdisc.data[3];
        jdisc.data[4] -= 1;
      }
      if(jdisc.data[1] > 11) {
        jdisc.data[1] = 11*2-jdisc.data[1];
        jdisc.data[3] = -jdisc.data[3];
        jdisc.data[4] -= 1;
      }
      if(jdisc.data[4] < 0) {
        idiscs.erase(idiscs.begin()+j, idiscs.begin()+(j+1));
      }
    }
  }

  // do damage

  float last_reward = model.hp[0] - model.hp[1];
  for(int i = 0; i < 2; ++i) {
    float hpx = model.hero_pos[i][0];
    float hpy = model.hero_pos[i][1];
    std::vector<disc_t>& idiscs = discs[i];
    for(auto jt = idiscs.begin(); jt != idiscs.end(); ++jt) {
      disc_t& jdisc = *jt;
      float dpx = jdisc.data[0];
      float dpy = jdisc.data[1];
      float damage = hypotf(hpx - dpx, hpy - dpy) < 1;
      model.hp[i] -= damage;
    }
  }
  float reward = model.hp[0] - model.hp[1] - last_reward;

  // decrement tick counter, check for end

  model.rem_ticks -= 1;
  bool done = model.rem_ticks <= 0 || model.hp[0] <= 0 || model.hp[1] <= 0;
  if(done) {
    if(model.hp[0] > 0 && model.hp[1] <= 0) {
      reward += 3;
    }
    if(model.hp[0] <= 0 && model.hp[1] > 0) {
      reward -= 3;
    }
  }

  // increment charge

  for(int i = 0; i < 2; ++i) {
    model.charge[i] += 1;
    if(model.charge[i] > 20)model.charge[i] = 20;
  }

  // generate observation

  struct lt_by_dist_t {
    float refx, refy;
    bool operator() (const disc_t& lhs, const disc_t& rhs) {
      return hypotf(lhs.data[0] - refx, lhs.data[1] - refy) < hypotf(rhs.data[0] - refx, rhs.data[1] - refy);
    }
  } lt_by_dist;

  for(int i = 0; i < 2; ++i) {
    lt_by_dist.refx = model.hero_pos[i][0];
    lt_by_dist.refy = model.hero_pos[i][1];
    std::sort(discs[i].begin(), discs[i].end(), lt_by_dist);
  }

  for(int i = 0; i < 2; ++i) {
    model.observation[i][0][0] = 1;
    model.observation[i][0][1] = model.hero_pos[i][0];
    model.observation[i][0][2] = model.hero_pos[i][1];
    model.observation[i][0][3] = 0;
    model.observation[i][0][4] = 0;
    model.observation[i][0][5] = model.hp[i];
    model.observation[i][0][6] = model.charge[i];
    std::vector<disc_t>& idiscs = discs[i];
    int jlim = idiscs.size();
    if(jlim > 10)jlim = 10;
    for(int j = 0; j < jlim; ++j) {
      disc_t& jdisc = idiscs[j];
      model.observation[i][j+1][0] = 1;
      model.observation[i][j+1][1] = jdisc.data[0];
      model.observation[i][j+1][2] = jdisc.data[1];
      model.observation[i][j+1][3] = jdisc.data[2];
      model.observation[i][j+1][4] = jdisc.data[3];
      model.observation[i][j+1][5] = jdisc.data[4];
      model.observation[i][j+1][6] = 0;
    }
    for(int j = jlim; j < 10; ++j) {
      for(int k = 0; k < 7; ++k) {
        model.observation[i][j+1][k] = 0;
      }
    }
  }

  // copy back

  model.reward = reward;
  model.done = done;

  for(int i = 0; i < 2; ++i) {
    std::vector<disc_t>& idiscs = discs[i];
    for(int j = 0; j < idiscs.size(); ++j) {
      disc_t& jdisc = idiscs[j];
      for(int k = 0; k < 5; ++k) {
        model.discs[i][j][k] = jdisc.data[k];
      }
    }
    for(int j = idiscs.size(); j < 50; ++j) {
      for(int k = 0; k < 5; ++k) {
        model.discs[i][j][k] = -999999;
      }
    }
  }

  for(int i = 0; i < 681; ++i) {
    PyObject* fbi = PyFloat_FromDouble((double)(model_f[i]));
    PyList_Append(nfbuffer_py, fbi);
  }

  return nfbuffer_py;
}

static PyMethodDef accel_methods[] = {
  {"new_match", accel_new_match, METH_VARARGS, "Start a new match, and run all the way up to the first query."},
  {"update", accel_update, METH_VARARGS, "Run an update step, up to the next query."},
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef accel_module = {
    PyModuleDef_HEAD_INIT,
    "ricochet_hetaro_accel",   /* name of module */
    "Faster low level implementation of the Ricochet game, for use with the Hetaro agents.", /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    accel_methods
};

PyMODINIT_FUNC
PyInit_ricochet_hetaro_accel(void)
{
    return PyModule_Create(&accel_module);
}
