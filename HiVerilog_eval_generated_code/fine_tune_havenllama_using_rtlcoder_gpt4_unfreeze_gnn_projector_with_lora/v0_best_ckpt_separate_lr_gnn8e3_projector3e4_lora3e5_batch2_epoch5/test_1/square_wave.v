module square_wave(
    input clk,
    input [7:0] freq,
    output wave_out
);
reg [7:0] count;
    
    always @(posedge clk) begin
        if (count == (freq - 1)) begin
            count <= 0;
        end else begin
            count <= count + 1;
        end
    end
    
    assign wave_out = (count == 0);
endmodule