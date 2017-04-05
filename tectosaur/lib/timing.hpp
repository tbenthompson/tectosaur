#pragma once
#include <chrono>
#include <iostream>

#define TIC\
    std::chrono::high_resolution_clock::time_point start =\
        std::chrono::high_resolution_clock::now();\
    int time_ms;
#define TIC2\
    start = std::chrono::high_resolution_clock::now();
#define TOC(name)\
    time_ms = std::chrono::duration_cast<std::chrono::microseconds>\
                (std::chrono::high_resolution_clock::now() - start).count();\
    std::cout << name << " took "\
              << time_ms\
              << "us.\n";

struct Timer {
    typedef std::chrono::high_resolution_clock::time_point Time;
    Time t_start;
    int time_us = 0;

    void start() {
        t_start = std::chrono::high_resolution_clock::now();
    }

    void stop() {
        time_us += std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - t_start
        ).count();
    }

    int get_time() {
        return time_us;
    }
};
