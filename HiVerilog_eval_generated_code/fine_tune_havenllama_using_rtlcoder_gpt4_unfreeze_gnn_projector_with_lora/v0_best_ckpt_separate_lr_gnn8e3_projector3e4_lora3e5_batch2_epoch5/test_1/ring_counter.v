module ring_counter (
    input wire clk,
    input wire reset,
    output reg [7:0] out
);

    
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            out <= 8'b00000_0001;
        end else begin
            out <= out;
        end
    end

    
    always @(posedge clk) begin
        if (out == 8'b11111_1111) begin
            out <= 8'b00000_0001;
        end else begin
            out <= out + 1;
        end
    end

endmodule