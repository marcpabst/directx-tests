// THIS IS A RUST PORT OF https://github.com/blurbusters/RefreshRateCalculator
// LICENSE - Apache-2.0
//
// Copyright 2014-2023 by Jerry Jongerius of DuckWare (https://www.duckware.com) - original code and algorithm
// Copyright 2017-2023 by Mark Rejhon of Blur Busters / TestUFO (https://www.testufo.com) - refactoring and improvements
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

pub struct RefreshRateCalculator {
    validatems: f64,
    tightgroupms: f64,
    mschange: f64,
    maxstore: usize,
    lowestvalidhz: f64,
    javascript_skip: i32,
    cycle_count: i32,
    n_skip_base: i32,
    n_skip: i32,
    t_update: f64,
    l0: f64,
    l1: f64,
    l2: f64,
    l3: f64,
    l4: f64,
    m_d: Vec<f64>,
    m_s: Vec<f64>,
    m_ms: f64,
    m_vi: String,
    m_tvsync: f64,
    m_summs: f64,
    m_nms: i32,
    m_nchange: i32,
}

impl RefreshRateCalculator {
    pub fn new() -> RefreshRateCalculator {
        let mut calculator = RefreshRateCalculator {
            validatems: 100.0,
            tightgroupms: 1.0,
            mschange: 1.0,
            maxstore: 60000,
            lowestvalidhz: 22.0,
            javascript_skip: 60,
            cycle_count: 0,
            n_skip_base: 0,
            n_skip: 0,
            t_update: 0.0,
            l0: 0.0,
            l1: 0.0,
            l2: 0.0,
            l3: 0.0,
            l4: 0.0,
            m_d: Vec::new(),
            m_s: Vec::new(),
            m_ms: 0.0,
            m_vi: String::new(),
            m_tvsync: 0.0,
            m_summs: 0.0,
            m_nms: 0,
            m_nchange: 0,
        };
        calculator.reset();
        calculator
    }

    pub fn restart_measuring(&mut self) {
        self.reset();
    }

    pub fn count_cycle(&mut self, current_time: f64) {
        self.add(current_time);
    }

    pub fn ignore_next_cycle(&mut self, cycles: i32) {
        self.n_skip = cycles;
    }

    pub fn get_minimum_frequency(&self) -> f64 {
        self.lowestvalidhz
    }

    pub fn set_minimum_frequency(&mut self, hz: f64) {
        self.lowestvalidhz = hz;
    }

    pub fn get_current_frequency(&self) -> f64 {
        self.calc()
    }

    pub fn get_filtered_cycle_timestamp(&self) -> Option<f64> {
        self.snap().map(|snap| snap.tvsync)
    }

    pub fn get_count(&self) -> i32 {
        self.cycle_count
    }

    pub fn get_raster_scanout_percentage(&self, current_time: f64) -> f64 {
        if let Some(snap) = self.snap() {
            let elapsed = current_time - snap.tvsync;
            let interval = 1000.0 / self.get_current_frequency();
            (elapsed % interval) / interval
        } else {
            0.0
        }
    }

    fn reset(&mut self) {
        self.cycle_count = 0;
        self.n_skip_base = 0;
        self.n_skip = 0;
        self.t_update = 0.0;
        self.l0 = 0.0;
        self.l1 = 0.0;
        self.l2 = 0.0;
        self.l3 = 0.0;
        self.l4 = 0.0;
        self.m_d.clear();
        self.m_s.clear();
        self.m_ms = 0.0;
        self.m_vi.clear();
        self.m_tvsync = 0.0;
        self.m_summs = 0.0;
        self.m_nms = 0;
        self.m_nchange = 0;
    }

    fn cut(&self, arr: &[f64]) -> Vec<f64> {
        arr.iter().step_by(2).copied().collect()
    }

    fn add(&mut self, t_frame: f64) {
        self.cycle_count += 1;
        self.l0 = self.l1;
        self.l1 = self.l2;
        self.l2 = self.l3;
        self.l3 = self.l4;
        self.l4 = t_frame;
        let grouping = f64::max(
            self.l4 - self.l3,
            f64::max(
                self.l3 - self.l2,
                f64::max(self.l2 - self.l1, self.l1 - self.l0),
            ),
        ) - f64::min(
            self.l4 - self.l3,
            f64::min(
                self.l3 - self.l2,
                f64::min(self.l2 - self.l1, self.l1 - self.l0),
            ),
        );

        if grouping < self.tightgroupms {
            if self.javascript_skip > 0 {
                self.javascript_skip -= 1;
            } else if self.n_skip > 0 {
                self.n_skip -= 1;
            } else {
                let avems = (self.l4 - self.l0) / 4.0;
                if avems < (1000.0 / self.lowestvalidhz) {
                    let b_hz_changed = self.m_nms > 10
                        && (avems - self.m_summs / self.m_nms as f64).abs() > self.mschange;
                    self.m_nchange = if b_hz_changed { self.m_nchange + 1 } else { 0 };
                    if !b_hz_changed {
                        self.m_summs += avems;
                        self.m_nms += 1;
                        if self.m_ms == 0.0 && self.m_nms > 30 {
                            self.m_ms = self.m_summs / self.m_nms as f64;
                        }
                    }

                    if self.m_nchange > 20 {
                        self.reset();
                    } else {
                        if self.m_d.len() >= self.maxstore {
                            self.m_d = self.cut(&self.m_d);
                            self.m_s = self.cut(&self.m_s);
                            self.n_skip_base = self.n_skip_base * 2 + 1;
                        }
                        self.m_d.push(self.l2);
                        self.m_s
                            .push((self.l0 + self.l1 + self.l2 + self.l3 + self.l4) / 5.0);
                        self.n_skip = self.n_skip_base;

                        if self.m_ms != 0.0 && t_frame - self.t_update > self.validatems {
                            self.t_update = t_frame;
                            self.m_vi = self.validate();
                        }
                    }
                }
            }
        }
    }

    fn calc(&self) -> f64 {
        if self.m_ms != 0.0 {
            1000.0 / self.m_ms
        } else {
            0.0
        }
    }

    fn validate(&mut self) -> String {
        let mut ret = String::new();
        if self.m_ms == 0.0 {
            return ret;
        }

        let (mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0);
        let mut x = 0;
        let n = self.m_d.len();
        for loop_idx in 0..n {
            x += if loop_idx > 0 {
                ((self.m_s[loop_idx] - self.m_s[loop_idx - 1]) / self.m_ms).round() as i32
            } else {
                0
            };
            let y = self.m_d[loop_idx] - self.m_d[0];
            sx += x as f64;
            sy += y;
            sxx += (x as f64) * (x as f64);
            sxy += (x as f64) * y;
        }

        let m = (n as f64 * sxy - sx * sy) / (n as f64 * sxx - sx * sx);
        let b = (sxx * sy - sx * sxy) / (n as f64 * sxx - sx * sx);
        self.m_ms = m;
        let tb = self.m_d[0] + b;

        let halfms = self.m_ms / 2.0;
        let mut max = 0.0;
        let mut min = 0.0;
        for loop_idx in 1..self.m_d.len() {
            let off = (self.m_d[loop_idx] - tb + halfms) % self.m_ms - halfms;
            min = f64::min(min, off);
            max = f64::max(max, off);
        }

        if max - min < halfms {
            self.m_tvsync = tb + min;
            ret = format!("drift=[{:.2}..{:.2}]", min, max);
        } else {
            self.reset();
        }
        ret
    }

    fn snap(&self) -> Option<Snap> {
        if !self.m_vi.is_empty() {
            Some(Snap {
                tvsync: self.m_tvsync,
                ms: self.m_ms,
            })
        } else {
            None
        }
    }
}

struct Snap {
    tvsync: f64,
    ms: f64,
}
