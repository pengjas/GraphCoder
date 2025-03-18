`timescale 1ns / 1ps

module tb_comparator_16bit;

    reg [15:0] a, b;
    wire eq, gt, lt;

    comparator_16bit UUT (
        .a(a),
        .b(b),
        .eq(eq),
        .gt(gt),
        .lt(lt)
    );

    initial begin
        // Test case 1: a == b
        a = 16'h1234; b = 16'h1234;
        #10;
        if (eq !== 1'b1 || gt !== 1'b0 || lt !== 1'b0) begin
            $display("===========Error in Test Case 1: a == b===========");
            $finish;
        end

        // Test case 2: a > b
        a = 16'hFFFF; b = 16'h0001;
        #10;
        if (eq !== 1'b0 || gt !== 1'b1 || lt !== 1'b0) begin
            $display("===========Error in Test Case 2: a > b===========");
            $finish;
        end

        // Test case 3: a < b
        a = 16'h0001; b = 16'hFFFF;
        #10;
        if (eq !== 1'b0 || gt !== 1'b0 || lt !== 1'b1) begin
            $display("===========Error in Test Case 3: a < b===========");
            $finish;
        end
        
        // Additional test cases can be added here

        $display("===========Your Design Passed===========");
        $finish;
    end

endmodule
