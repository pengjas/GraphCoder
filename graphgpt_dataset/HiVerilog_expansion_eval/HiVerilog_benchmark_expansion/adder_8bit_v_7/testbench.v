`timescale 1ns / 1ps

module testbench_subtractor_8bit;

    reg [7:0] a, b;
    reg bin;
    wire [7:0] diff;
    wire bout;
    integer i;

    // Instantiate the Unit Under Test (UUT)
    subtractor_8bit uut (
        .a(a), 
        .b(b), 
        .bin(bin), 
        .diff(diff), 
        .bout(bout)
    );

    initial begin
        // Initialize Inputs
        a = 0;
        b = 0;
        bin = 0;
        
        // Wait 100 ns for global reset to finish
        #100;
        
        // Add stimulus here
        // Test Case 1: Simple subtraction with no borrow
        a = 8'd100; b = 8'd50; bin = 1'b0;
        #10; 
        check_result(8'd50, 1'b0);

        // Test Case 2: Subtraction with borrow-in
        a = 8'd50; b = 8'd100; bin = 1'b0;
        #10; 
        check_result(8'd206, 1'b1);  // 50 - 100 = -50 -> 0b11001110 with borrow
        
        // Test Case 3: No operation difference
        a = 8'd123; b = 8'd123; bin = 1'b0;
        #10; 
        check_result(8'd0, 1'b0);

        // Test Case 4: Full range check with borrow out
        a = 8'h00; b = 8'hFF; bin = 1'b1;
        #10;
        check_result(8'h00, 1'b1);
        
        // Test Case 5: Random values with no borrow
        a = 8'd15; b = 8'd7; bin = 1'b0;
        #10;
        check_result(8'd8, 1'b0);
    end
    
    task check_result;
        input [7:0] expected_diff;
        input expected_bout;
        begin
            if (diff !== expected_diff || bout !== expected_bout) begin
                $display("===========Error=========== at time %t", $time);
                $display("Test failed: Expected diff = %b, Actual diff = %b", expected_diff, diff);
                $display("Expected bout = %b, Actual bout = %b", expected_bout, bout);
                $finish;
            end
            else begin
                $display("Test Passed: A = %d, B = %d, Bin = %b, Diff = %d, Bout = %b at time %t", a, b, bin, diff, bout, $time);
            end
        end
    endtask

    initial begin
        // Finish the simulation after a delay to run all tests
        #500;
        $display("===========Your Design Passed===========");
        $finish;
    end

endmodule
