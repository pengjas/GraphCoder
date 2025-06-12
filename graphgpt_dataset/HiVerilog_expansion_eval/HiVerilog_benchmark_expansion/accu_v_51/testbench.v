`timescale 1ns / 1ps

module tb_binary_up_counter;

    // Inputs
    reg clk;
    reg rst;
    reg en;

    // Outputs
    wire [3:0] count_out;
    wire overflow;

    // Instantiate the Unit Under Test (UUT)
    binary_up_counter uut (
        .clk(clk),
        .rst(rst),
        .en(en),
        .count_out(count_out),
        .overflow(overflow)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // Clock with period of 10ns
    end

    // Test cases and checking results
    initial begin
        // Initialize Inputs
        rst = 0;
        en = 0;

        // Wait 100 ns for global reset
        #100;

        // Case 1: Reset the counter
        rst = 1; #10;
        rst = 0; #10;
        if (count_out !== 4'b0000) begin
            $display("===========Error: Reset not working===========");
            $finish;
        end

        // Case 2: Enable and check counter increment
        en = 1;
        repeat(16) begin
            @(posedge clk);
            if (count_out === 4'b1111 && overflow !== 1'b1) begin
                $display("===========Error: Overflow not detected===========");
                $finish;
            end
        end
        en = 0;
        
        // Case 3: Ensure counter does not increment when enable is low
        @(posedge clk);
        if (count_out !== 4'b1111) begin
            $display("===========Error: Counter incrementing when enable is low===========");
            $finish;
        end

        // Case 4: Reset after operation
        rst = 1; #10;
        rst = 0; #10;
        if (count_out !== 4'b0000) begin
            $display("===========Error: Reset after operation not working===========");
            $finish;
        end

        // If all tests passed
        $display("===========Your Design Passed===========");
        $finish;
    end

endmodule
