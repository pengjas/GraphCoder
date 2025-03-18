`timescale 1ns / 1ps

module tb_subtractor_8bit;

    reg [7:0] a, b;
    reg bin;
    wire [7:0] diff;
    wire bout;

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
        a = 0; b = 0; bin = 0;
        
        // Wait 100 ns for global reset to finish
        #100;
        
        // Test case 1: Simple subtraction with no borrow
        a = 8'd50; b = 8'd20; bin = 1'b0;
        #10; // Wait for propagation
        check_results(8'd30, 1'b0);
        
        // Test case 2: Subtraction causing borrow
        a = 8'd20; b = 8'd50; bin = 1'b0;
        #10; // Wait for propagation
        check_results(8'd226, 1'b1); // 20 - 50 = -30 => 256 - 30 = 226 with borrow

        // Test case 3: Check with borrow in
        a = 8'd15; b = 8'd25; bin = 1'b1;
        #10; // Wait for propagation
        check_results(8'd245, 1'b1); // 15 - 25 - 1 = -11 => 256 - 11 = 245 with borrow

        // Test case 4: zero subtract zero
        a = 8'd0; b = 8'd0; bin = 0;
        #10;
        check_results(8'd0, 1'b0);

        // Check completed tests
        $display("===========Your Design Passed===========");
        $finish;
    end

    task check_results;
        input [7:0] expected_diff;
        input expected_bout;
        begin
            if (diff !== expected_diff || bout !== expected_bout) begin
                $display("===========Error in Test Case===========");
                $display("Expected diff: %d, Output diff: %d", expected_diff, diff);
                $display("Expected bout: %b, Output bout: %b", expected_bout, bout);
                $finish;
            end
        end
    endtask
    
endmodule
