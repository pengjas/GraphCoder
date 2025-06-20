`timescale 1ns/1ps

module tb_pwm_gen;

    // Inputs
    reg CLK_in;
    reg RST;

    // Outputs
    wire PWM_25;
    wire PWM_50;
    wire PWM_75;

    // Instantiate the Unit Under Test (UUT)
    pwm_gen uut (
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

    // Test case variables
    integer i;
    integer error_count = 0;
    integer cycles_25 = 0;
    integer cycles_50 = 0;
    integer cycles_75 = 0;
    
    // Generate Reset
    initial begin
        // Initialize Inputs
        RST = 1;

        // Wait 100 ns for global reset to finish
        #100;
        
        // Release reset
        RST = 0;

        // Count PWM pulses for 1000 clock cycles
        for (i = 0; i < 1000; i = i + 1) begin
            @(posedge CLK_in);
            cycles_25 = cycles_25 + PWM_25;
            cycles_50 = cycles_50 + PWM_50;
            cycles_75 = cycles_75 + PWM_75;
        end

        // Check results
        if (cycles_25 !== 250) begin
            $display("Error: PWM_25 duty cycle mismatch: Expected 250, got %d", cycles_25);
            error_count = error_count + 1;
        end
        if (cycles_50 !== 500) begin
            $display("Error: PWM_50 duty cycle mismatch: Expected 500, got %d", cycles_50);
            error_count = error_count + 1;
        end
        if (cycles_75 !== 750) begin
            $display("Error: PWM_75 duty cycle mismatch: Expected 750, got %d", cycles_75);
            error_count = error_count + 1;
        end

        // Display result
        if (error_count === 0) begin
            $display("===========Your Design Passed===========");
        end else begin
            $display("===========Error===========");
        end

        // Finish simulation
        $finish;
    end

endmodule
