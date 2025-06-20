`timescale 1ns / 1ps

module tb_pwm_generator;

    reg CLK_in;
    reg RST;
    wire PWM_25;
    wire PWM_50;
    wire PWM_75;

    // Instance of pwm_generator module
    pwm_generator UUT (
        .CLK_in(CLK_in),
        .RST(RST),
        .PWM_25(PWM_25),
        .PWM_50(PWM_50),
        .PWM_75(PWM_75)
    );

    // Clock generation
    initial begin
        CLK_in = 0;
        forever #5 CLK_in = ~CLK_in; // 100 MHz Clock
    end

    // Reset generation
    initial begin
        RST = 1;
        #100; // Assert reset for 100ns
        RST = 0;
    end

    // Test cases and result checking
    integer pwm_25_count, pwm_50_count, pwm_75_count, total_cycles;
    initial begin
        pwm_25_count = 0;
        pwm_50_count = 0;
        pwm_75_count = 0;
        total_cycles = 0;
        
        // Monitor PWM outputs for a certain number of clock cycles
        #200; // Wait for reset to deassert and system to stabilize
        while (total_cycles < 1000) begin
            @(posedge CLK_in);
            pwm_25_count = pwm_25_count + PWM_25;
            pwm_50_count = pwm_50_count + PWM_50;
            pwm_75_count = pwm_75_count + PWM_75;
            total_cycles = total_cycles + 1;
        end

        // Check PWM_25 (25% duty cycle)
        if (pwm_25_count < 240 || pwm_25_count > 260) begin
            $display("===========Error===========: PWM_25 failed. Count=%d", pwm_25_count);
            $finish;
        end

        // Check PWM_50 (50% duty cycle)
        if (pwm_50_count < 490 || pwm_50_count > 510) begin
            $display("===========Error===========: PWM_50 failed. Count=%d", pwm_50_count);
            $finish;
        end

        // Check PWM_75 (75% duty cycle)
        if (pwm_75_count < 740 || pwm_75_count > 760) begin
            $display("===========Error===========: PWM_75 failed. Count=%d", pwm_75_count);
            $finish;
        end

        // If no errors have been flagged, pass the test
        $display("===========Your Design Passed===========");
    end
endmodule
