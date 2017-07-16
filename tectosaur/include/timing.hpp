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

    Timer() {
        restart(); 
    }

    void restart() {
        t_start = std::chrono::high_resolution_clock::now();
    }

    void report(std::string name) {
        int time_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - t_start
        ).count();
        std::string text = name + " took " + std::to_string(time_us) + "us";
        std::cout << text << std::endl;
        restart();
    }
};
